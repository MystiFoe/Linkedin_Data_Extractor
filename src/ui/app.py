import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time
import streamlit as st
import json
import pandas as pd
import io
import os
from src.api.linkedin_api import LinkedInAPI
from src.data.data_processor import DataProcessor
from src.logger import app_logger
from src.ui.linkedin_comment_page import linkedin_comment_page

class LinkedInExtractorApp:
    """Main Streamlit application class for LinkedIn Data Extractor"""
    
    def __init__(self):
        """Initialize the application"""
        self.linkedin_api = LinkedInAPI()
        self.data_processor = DataProcessor()
        app_logger.debug("Initializing LinkedIn Extractor App")
        
    def setup_page(self):
        """Set up the Streamlit page with title and custom styling"""
        st.set_page_config(
            page_title="Allied LinkedIn Data Extractor",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        st.markdown("""
        <style>
        body { background: #f4f8fb; }
        .main-header { font-size: 2.7rem; color: #0A66C2; text-align: center; font-weight: 900; margin-top: 1.5rem; margin-bottom: 0.5rem; letter-spacing: 1px; }
        .section-header { font-size: 1.35rem; color: #0A66C2; font-weight: 700; margin-top: 1.5rem; margin-bottom: 0.7rem; border-left: 5px solid #0A66C2; padding-left: 0.7rem; }
        .card { background: #fff; border-radius: 16px; box-shadow: 0 6px 32px rgba(10,102,194,0.13); padding: 2.2rem 2.7rem 1.7rem 2.7rem; margin-bottom: 2.2rem; }
        .stButton > button, .sidebar-btn { background: linear-gradient(90deg, #0A66C2 0%, #0056a3 100%); color: #fff; font-weight: 700; border: none; border-radius: 7px; padding: 0.8rem 1.7rem; font-size: 1.13rem; margin: 0.3rem 0.5rem 0.3rem 0; box-shadow: 0 2px 8px rgba(10,102,194,0.10); transition: background 0.2s; }
        .stButton > button:hover, .sidebar-btn.selected { background: linear-gradient(90deg, #0056a3 0%, #0A66C2 100%); }
        .sidebar-title { color: #0A66C2; font-size: 1.25rem; font-weight: 800; margin-bottom: 1.2rem; }
        .stTextInput > div > input, .stTextArea > div > textarea { background: #f4f8fb; border-radius: 7px; border: 1.7px solid #c7e0fc; font-size: 1.12rem; padding: 0.8rem 1.1rem; }
        .stDataFrame, .stTable { border-radius: 10px; box-shadow: 0 2px 10px rgba(10,102,194,0.09); margin-bottom: 1.7rem; }
        .stDownloadButton > button { margin-top: 1.3rem; }
        .stCheckbox > label { font-size: 1.12rem; color: #0A66C2; font-weight: 600; }
        .stSpinner { color: #0A66C2 !important; }
        .success-msg, .error-msg { font-size: 1.12rem; }
        hr { border: none; border-top: 1.7px solid #e3eaf3; margin: 1.7rem 0; }
        .stExpanderHeader { font-size: 1.08rem !important; font-weight: 600 !important; color: #0A66C2 !important; }
        .stExpanderContent { background: #f8fbff !important; }
        .stSidebar { background: #e8f0fe !important; }
        </style>
        """, unsafe_allow_html=True)
        st.markdown("<div class='main-header'>Allied LinkedIn Data Extractor</div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; margin-bottom: 2rem; font-size:1.13rem; color:#444;'>Extract LinkedIn data with Allied Worldwide's automation platform</p>", unsafe_allow_html=True)

    def display_navigation(self):
        """Display visually appealing sidebar navigation with styled buttons"""
        st.sidebar.markdown("<div class='sidebar-title'>Navigation</div>", unsafe_allow_html=True)
        pages = [
            ("Keyword Search", "Search LinkedIn posts by keywords"),
            ("Post Extraction", "Extract data from a specific LinkedIn post"),
            ("Profile Extraction", "Extract data from a LinkedIn profile"),
            ("Company Extraction", "Extract data from a LinkedIn company page"),
            ("Decision-Maker Pipeline", "End-to-end workflow: search, extract, merge, filter, export"),
            ("LinkedIn Comment Generator", "Generate comments for LinkedIn posts using AI"),
            ("AI Comment Generator (Blended)", "Generate unique, meaningful LinkedIn comments using the new persona-blended logic")
        ]
        selected_page = st.session_state.get("selected_page", pages[0][0])
        for page, tooltip in pages:
            btn = st.sidebar.button(
                page,
                key=f"nav_{page}",
                help=tooltip,
                use_container_width=True
            )
            if btn:
                st.session_state["selected_page"] = page
                selected_page = page
            # Highlight selected
            st.markdown(f"""
                <style>
                [data-testid="stSidebar"] button[key="nav_{page}"] {{
                    {'background: linear-gradient(90deg, #0A66C2 0%, #0056a3 100%); color: white;' if selected_page == page else ''}
                }}
                </style>
            """, unsafe_allow_html=True)
        st.sidebar.markdown("---")
        st.sidebar.markdown("### About")
        st.sidebar.info("This tool extracts LinkedIn data using Allied Worldwide's automation platform.")
        return st.session_state.get("selected_page", pages[0][0])
    
    def _get_automation_and_account(self, automation_label):
        """Helper to get automationId and connectedAccountId for a given automation label"""
        # Use the correct LinkedIn platform ID from TexAU
        platform_id = "622f03eb770f6bba0b8facaa"  # LinkedIn
        # Use the correct automation ID for LinkedIn Post Scraper
        if automation_label.lower() == "post extraction":
            automation_id = "63fdd06c82e9647288a2d925"  # LinkedIn Post Scraper
        elif automation_label.lower() == "profile extraction":
            automation_id = "63f48ee97022e05c116fc798"  # LinkedIn Profile Scraper
        else:
            automations = self.linkedin_api.get_automations(platform_id)
            automation_id = None
            for a in automations.get("data", []):
                if automation_label.lower() in a.get("label", "").lower():
                    automation_id = a["id"]
                    break
        # Use the actual connected LinkedIn account ID
        connected_account_id = "67ffa43877d8e1e658850bc3"
        return automation_id, connected_account_id

    def remove_empty_columns(self, df):
        # Remove columns where all values are empty (NaN or empty string)
        return df.dropna(axis=1, how='all').loc[:, ~(df == '').all(axis=0)]

    def expand_profiles_to_df(self, profiles):
        """
        Robustly expand a list of profile results (which may be JSON strings or dicts, or lists of dicts)
        into a columnar DataFrame. Handles nested lists and mixed types.
        """
        if not profiles:
            return pd.DataFrame()
        expanded = []
        for p in profiles:
            if isinstance(p, str):
                try:
                    obj = json.loads(p)
                except Exception:
                    continue
            else:
                obj = p
            if isinstance(obj, list):
                expanded.extend(obj)
            else:
                expanded.append(obj)
        return pd.json_normalize(expanded)

    def keyword_search_page(self):
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>🔍 Search LinkedIn Posts by Keywords</div>", unsafe_allow_html=True)
        st.caption("Find posts by keyword or LinkedIn search URL. Use filters for more control.")
        search_input = st.text_input(
            "Keyword or LinkedIn Search URL",
            placeholder="e.g., Marketing or https://www.linkedin.com/search/results/content/?keywords=Marketing",
            help="Enter a keyword or LinkedIn search URL."
        )
        with st.expander("Show Filters (optional)", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                start_time = st.selectbox(
                    "Event Start Time",
                    ["", "PAST 24H", "PAST WEEK", "PAST MONTH"],
                    format_func=lambda x: x if x else "-- None --"
                )
                sort_by = st.selectbox(
                    "Sort By",
                    ["", "DATE POSTED", "RELEVANCE"],
                    format_func=lambda x: x if x else "-- None --"
                )
            with col2:
                posted_by = st.selectbox(
                    "Posted By",
                    ["", "1st CONNECTION", "ME", "PEOPLE YOU FOLLOW"],
                    format_func=lambda x: x if x else "-- None --"
                )
                extract_limit = st.number_input(
                    "Post Extraction Limit (Max. 2500)",
                    min_value=1, max_value=2500, value=10, step=1, format="%d"
                )
        extract_btn = st.button("Extract Posts", help="Start extraction based on your search and filters.")
        if extract_btn:
            if not search_input:
                st.error("Please enter a keyword or LinkedIn search URL.")
            else:
                with st.spinner("Extracting posts..."):
                    automation_id = "64099c6e0936e46db5d76f4c"
                    _, connected_account_id = self._get_automation_and_account("keyword search")
                    api_inputs = {"liPostSearchUrl": search_input}
                    if start_time:
                        api_inputs["startTime"] = {"PAST 24H": "past-24h", "PAST WEEK": "past-week", "PAST MONTH": "past-month"}[start_time]
                    if sort_by:
                        api_inputs["sortBy"] = {"DATE POSTED": "date_posted", "RELEVANCE": "relevance"}[sort_by]
                    if posted_by:
                        api_inputs["postedBy"] = {"1st CONNECTION": "first", "ME": "me", "PEOPLE YOU FOLLOW": "following"}[posted_by]
                    if extract_limit:
                        api_inputs["maxCountPostSearch"] = int(extract_limit)
                    result = self.linkedin_api.run_automation(
                        name="Post Search Export",
                        description="Export LinkedIn posts by keywords",
                        automation_id=automation_id,
                        connected_account_id=connected_account_id,
                        timezone="Asia/Kolkata",
                        inputs=api_inputs
                    )
                    data = result.get("data", {})
                    execution_id = data.get("id") or data.get("workflowId")
                    final_result = None
                    if execution_id:
                        for _ in range(120):
                            final_result = self.linkedin_api.get_execution_result(execution_id)
                            if final_result.get("data"):
                                break
                            time.sleep(1)
                    if final_result and "data" in final_result:
                        df = pd.json_normalize(final_result["data"])
                        df = self.remove_empty_columns(df)
                        st.success(f"Found {len(df)} posts.")
                        st.dataframe(df, use_container_width=True)
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                            if not df.empty:
                                df.to_excel(writer, sheet_name="keyword_posts", index=False)
                        excel_buffer.seek(0)
                        st.download_button(
                            label="Download as Excel",
                            data=excel_buffer.getvalue(),
                            file_name="linkedin_keyword_posts.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    else:
                        st.warning("No posts found matching the search.")
        st.markdown("</div>", unsafe_allow_html=True)

    def post_extraction_page(self):
        """Display post extraction page"""
        st.markdown("<h2 class='section-header'>Extract LinkedIn Post Data</h2>", unsafe_allow_html=True)
        with st.container():
            post_url = st.text_input("Enter LinkedIn Post URL", placeholder="https://www.linkedin.com/posts/...")
            extract_likers = st.checkbox("Also extract post likers (optional)")
            extract_comments = st.checkbox("Also extract post comments (optional)")
            if post_url or st.session_state.get('post_extraction_result'):
                st.markdown("<div class='data-card'>", unsafe_allow_html=True)
            if st.button("Extract Post Data"):
                if post_url and "linkedin.com" in post_url:
                    try:
                        app_logger.info("Extracting data for post: {}", post_url)
                        with st.spinner("Extracting post data..."):
                            # 1. Extract post content
                            automation_id, connected_account_id = self._get_automation_and_account("post extraction")
                            result = self.linkedin_api.extract_post_data(post_url, automation_id, connected_account_id)
                            if result and "data" in result:
                                dfs = self.data_processor.convert_to_dataframe(result, "post")
                                for key in dfs:
                                    dfs[key] = self.remove_empty_columns(dfs[key])
                                st.subheader("Post Information")
                                st.dataframe(dfs["post"], use_container_width=True)
                                if not dfs["reactors"].empty:
                                    st.subheader(f"Reactors ({len(dfs['reactors'])})")
                                    st.dataframe(dfs["reactors"], use_container_width=True)
                                if not dfs["commenters"].empty:
                                    st.subheader(f"Commenters ({len(dfs['commenters'])})")
                                    st.dataframe(dfs["commenters"], use_container_width=True)
                                # 2. Optionally extract likers
                                if extract_likers:
                                    with st.spinner("Extracting post likers (this may take a while)..."):
                                        likers_automation_id = "63fc575f7022e05c11bba145"  # LinkedIn Post Likers Export
                                        likers_result = self.linkedin_api.run_automation(
                                            name="Post Likers Export",
                                            description="Export LinkedIn post likers",
                                            automation_id=likers_automation_id,
                                            connected_account_id=connected_account_id,
                                            timezone="Asia/Kolkata",
                                            inputs={"liPostUrl": post_url}
                                        )
                                        data = likers_result.get("data", {})
                                        execution_id = data.get("id") or data.get("workflowId")
                                        likers_final_result = None
                                        if execution_id:
                                            for _ in range(60):
                                                likers_final_result = self.linkedin_api.get_execution_result(execution_id)
                                                if likers_final_result.get("data"):
                                                    break
                                                time.sleep(1)
                                        likers_df = pd.DataFrame()
                                        if likers_final_result and "data" in likers_final_result:
                                            likers_data = likers_final_result["data"]
                                            if isinstance(likers_data, list):
                                                likers_df = pd.json_normalize(likers_data)
                                            elif isinstance(likers_data, dict):
                                                likers_df = pd.json_normalize([likers_data])
                                        dfs["likers"] = self.remove_empty_columns(likers_df)
                                        if not likers_df.empty:
                                            st.subheader(f"Likers ({len(likers_df)})")
                                            st.dataframe(likers_df, use_container_width=True)
                                # 3. Optionally extract comments
                                if extract_comments:
                                    with st.spinner("Extracting post comments (this may take a while)..."):
                                        comments_automation_id = "63fc8cd27022e05c113c3c73"  # LinkedIn Comments Scraper
                                        comments_result = self.linkedin_api.run_automation(
                                            name="Comments Export",
                                            description="Export LinkedIn post comments",
                                            automation_id=comments_automation_id,
                                            connected_account_id=connected_account_id,
                                            timezone="Asia/Kolkata",
                                            inputs={"liPostUrl": post_url}
                                        )
                                        data = comments_result.get("data", {})
                                        execution_id = data.get("id") or data.get("workflowId")
                                        comments_final_result = None
                                        if execution_id:
                                            for _ in range(60):
                                                comments_final_result = self.linkedin_api.get_execution_result(execution_id)
                                                if comments_final_result.get("data"):
                                                    break
                                                time.sleep(1)
                                        comments_df = pd.DataFrame()
                                        if comments_final_result and "data" in comments_final_result:
                                            comments_data = comments_final_result["data"]
                                            if isinstance(comments_data, list):
                                                comments_df = pd.json_normalize(comments_data)
                                            elif isinstance(comments_data, dict):
                                                comments_df = pd.json_normalize([comments_data])
                                        dfs["comments_export"] = self.remove_empty_columns(comments_df)
                                        if not comments_df.empty:
                                            st.subheader(f"Comments Export ({len(comments_df)})")
                                            st.dataframe(comments_df, use_container_width=True)
                                # Download as Excel (multi-sheet)
                                excel_buffer = io.BytesIO()
                                with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                                    for sheet_name, df in dfs.items():
                                        if not df.empty:
                                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                                excel_buffer.seek(0)
                                if st.download_button(
                                    label="Download as Excel",
                                    data=excel_buffer.getvalue(),
                                    file_name=f"linkedin_post_data.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                ):
                                    filepath = self.data_processor.export_to_excel(dfs, "post")
                                    st.markdown(f"<p class='success-msg'>File saved to: {filepath}</p>", unsafe_allow_html=True)
                            else:
                                st.warning("No data found for this post URL.")
                    except Exception as e:
                        app_logger.error("Error in post extraction: {}", str(e))
                        st.error(f"An error occurred: {str(e)}")
                else:
                    st.warning("Please enter a valid LinkedIn post URL.")
            if post_url or st.session_state.get('post_extraction_result'):
                st.markdown("</div>", unsafe_allow_html=True)

    def profile_extraction_page(self):
        """Display profile extraction page"""
        st.markdown("<h2 class='section-header'>Extract LinkedIn Profile Data</h2>", unsafe_allow_html=True)
        with st.container():
            profile_url = st.text_input("Enter LinkedIn Profile URL", placeholder="https://www.linkedin.com/in/...")
            if profile_url or st.session_state.get('profile_extraction_result'):
                st.markdown("<div class='data-card'>", unsafe_allow_html=True)
            extract_activity = st.checkbox("Also extract profile's latest activities (optional)")
            activity_limit = None
            if extract_activity:
                activity_limit = st.number_input(
                    "Activity Extraction Limit (Max. 1000)",
                    min_value=1, max_value=1000, value=20, step=1, format="%d"
                )
            extract_posts = st.checkbox("Also extract profile's posts (optional)")
            posts_limit = None
            if extract_posts:
                posts_limit = st.number_input(
                    "Posts Extraction Limit (Max. 1000)",
                    min_value=1, max_value=1000, value=20, step=1, format="%d"
                )
            extract_clicked = st.button("Extract Profile Data")
            activity_df = pd.DataFrame()
            posts_df = pd.DataFrame()
            if extract_clicked:
                if profile_url and "linkedin.com/in/" in profile_url:
                    try:
                        app_logger.info("Extracting data for profile: {}", profile_url)
                        with st.spinner("Extracting profile data..."):
                            # Use correct input key for LinkedIn Profile Scraper
                            automation_id, connected_account_id = self._get_automation_and_account("profile extraction")
                            result = self.linkedin_api.run_automation(
                                name="Profile Extraction",
                                description="Extract LinkedIn profile data",
                                automation_id=automation_id,
                                connected_account_id=connected_account_id,
                                timezone="Asia/Kolkata",
                                inputs={"liProfileUrl": profile_url}
                            )
                            # Get execution id from result
                            data = result.get("data", {})
                            execution_id = data.get("id") or data.get("workflowId")
                            final_result = None
                            if execution_id:
                                for _ in range(60):
                                    final_result = self.linkedin_api.get_execution_result(execution_id)
                                    if final_result.get("data"):
                                        break
                                    time.sleep(1)
                            all_dfs = {}
                            if final_result and "data" in final_result:
                                df = self.data_processor.convert_to_dataframe(final_result, "profile")
                                df = self.remove_empty_columns(df)
                                st.subheader("Profile Information")
                                st.dataframe(df, use_container_width=True)
                                # Format and display additional profile sections nicely
                                profile_data = final_result.get("data", {})
                                all_dfs["profile"] = df
                                if "experiences" in profile_data:
                                    st.subheader("Experiences")
                                    exp_df = pd.json_normalize(profile_data["experiences"])
                                    exp_df = self.remove_empty_columns(exp_df)
                                    st.dataframe(exp_df, use_container_width=True)
                                    all_dfs["experiences"] = exp_df
                                if "education" in profile_data:
                                    st.subheader("Education")
                                    edu_df = pd.json_normalize(profile_data["education"])
                                    edu_df = self.remove_empty_columns(edu_df)
                                    st.dataframe(edu_df, use_container_width=True)
                                    all_dfs["education"] = edu_df
                                if "skills" in profile_data:
                                    st.subheader("Skills")
                                    skills_df = pd.DataFrame(profile_data["skills"], columns=["Skill"])
                                    skills_df = self.remove_empty_columns(skills_df)
                                    st.dataframe(skills_df, use_container_width=True)
                                    all_dfs["skills"] = skills_df
                        # Optional: Extract profile activity if checkbox is selected
                        if extract_activity:
                            with st.spinner("Extracting profile's latest activities (this may take a while)..."):
                                activity_automation_id = "63f5bf1d7022e05c1119cff2"  # LinkedIn Profile Activity Export
                                activity_inputs = {"liProfileUrl": profile_url}
                                if activity_limit:
                                    activity_inputs["maxCount"] = int(activity_limit)
                                activity_result = self.linkedin_api.run_automation(
                                    name="Profile Activity Export",
                                    description="Export LinkedIn profile activity",
                                    automation_id=activity_automation_id,
                                    connected_account_id=connected_account_id,
                                    timezone="Asia/Kolkata",
                                    inputs=activity_inputs
                                )
                                activity_data = activity_result.get("data", {})
                                activity_execution_id = activity_data.get("id") or activity_data.get("workflowId")
                                activity_final_result = None
                                if activity_execution_id:
                                    for _ in range(600):  # wait up to 10 minutes
                                        activity_final_result = self.linkedin_api.get_execution_result(activity_execution_id)
                                        if activity_final_result.get("data"):
                                            break
                                        time.sleep(1)
                                if activity_final_result and "data" in activity_final_result:
                                    # TexAU sometimes returns activity data under a nested key, handle both cases
                                    activity_data = activity_final_result["data"]
                                    if isinstance(activity_data, list):
                                        activity_df = pd.json_normalize(activity_data)
                                    elif isinstance(activity_data, dict):
                                        for v in activity_data.values():
                                            if isinstance(v, list):
                                                activity_df = pd.json_normalize(v)
                                                break
                                        else:
                                            activity_df = pd.json_normalize([activity_data])
                                    activity_df = self.remove_empty_columns(activity_df)
                                    if not activity_df.empty:
                                        st.subheader(f"Profile Activity ({len(activity_df)})")
                                        st.dataframe(activity_df, use_container_width=True)
                                        all_dfs["profile_activity"] = activity_df
                        # Optional: Extract profile posts if checkbox is selected
                        if extract_posts:
                            with st.spinner("Extracting profile's posts (this may take a while)..."):
                                posts_automation_id = "649425e10f7b435e858547c2"  # LinkedIn Profile Posts Export
                                posts_inputs = {"liProfileUrl": profile_url}
                                if posts_limit:
                                    posts_inputs["maxCount"] = int(posts_limit)
                                posts_result = self.linkedin_api.run_automation(
                                    name="Profile Posts Export",
                                    description="Export LinkedIn profile posts",
                                    automation_id=posts_automation_id,
                                    connected_account_id=connected_account_id,
                                    timezone="Asia/Kolkata",
                                    inputs=posts_inputs
                                )
                                data = posts_result.get("data", {})
                                execution_id = data.get("id") or data.get("workflowId")
                                posts_final_result = None
                                if execution_id:
                                    for _ in range(120):
                                        posts_final_result = self.linkedin_api.get_execution_result(execution_id)
                                        if posts_final_result.get("data"):
                                            break
                                        time.sleep(1)
                                if posts_final_result and "data" in posts_final_result:
                                    posts_data = posts_final_result["data"]
                                    if isinstance(posts_data, list):
                                        posts_df = pd.json_normalize(posts_data)
                                    elif isinstance(posts_data, dict):
                                        for v in posts_data.values():
                                            if isinstance(v, list):
                                                posts_df = pd.json_normalize(v)
                                                break
                                        else:
                                            posts_df = pd.json_normalize([posts_data])
                                    posts_df = self.remove_empty_columns(posts_df)
                                    if not posts_df.empty:
                                        st.subheader(f"Profile Posts ({len(posts_df)})")
                                        st.dataframe(posts_df, use_container_width=True)
                                        all_dfs["profile_posts"] = posts_df
                        # Download as Excel (multi-sheet)
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                            for sheet_name, df in all_dfs.items():
                                if not df.empty:
                                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                        excel_buffer.seek(0)
                        if st.download_button(
                            label="Download as Excel",
                            data=excel_buffer.getvalue(),
                            file_name=f"linkedin_profile_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        ):
                            filepath = self.data_processor.export_to_excel(all_dfs, "profile")
                            st.markdown(f"<p class='success-msg'>File saved to: {filepath}</p>", unsafe_allow_html=True)
                    except Exception as e:
                        app_logger.error("Error in profile extraction: {}", str(e))
                        st.error(f"An error occurred: {str(e)}")
                else:
                    st.warning("Please enter a valid LinkedIn profile URL.")
            if profile_url or st.session_state.get('profile_extraction_result'):
                st.markdown("</div>", unsafe_allow_html=True)

    def company_extraction_page(self):
        """Display company extraction page"""
        st.markdown("<h2 class='section-header'>Extract LinkedIn Company Data</h2>", unsafe_allow_html=True)
        with st.container():
            company_url = st.text_input("Enter LinkedIn Company URL", placeholder="https://www.linkedin.com/company/...")
            extract_insights = st.checkbox("Also extract company insights (optional)")
            if company_url or st.session_state.get('company_extraction_result'):
                st.markdown("<div class='data-card'>", unsafe_allow_html=True)
            if st.button("Extract Company Data"):
                if company_url and "linkedin.com/company/" in company_url:
                    try:
                        app_logger.info("Extracting data for company: {}", company_url)
                        with st.spinner("Extracting company data..."):
                            # Use correct automation ID and input key for LinkedIn Company Scraper
                            automation_id = "63f742037022e05c11a9440e"  # LinkedIn Company Scraper
                            connected_account_id = self._get_automation_and_account("company extraction")[1]
                            result = self.linkedin_api.run_automation(
                                name="Company Extraction",
                                description="Extract LinkedIn company data",
                                automation_id=automation_id,
                                connected_account_id=connected_account_id,
                                timezone="Asia/Kolkata", 
                                inputs={"liCompanyUrl": company_url}
                            )
                            data = result.get("data", {})
                            execution_id = data.get("id") or data.get("workflowId")
                            final_result = None
                            if execution_id:
                                for _ in range(120):
                                    final_result = self.linkedin_api.get_execution_result(execution_id)
                                    if final_result.get("data"):
                                        break
                                    time.sleep(1)
                            dfs = {}
                            if final_result and "data" in final_result:
                                dfs = self.data_processor.convert_to_dataframe(final_result, "company")
                                for key in dfs:
                                    dfs[key] = self.remove_empty_columns(dfs[key])
                                st.subheader("Company Information")
                                st.dataframe(dfs["company"], use_container_width=True)
                                if not dfs["personnel"].empty:
                                    st.subheader("Key Personnel")
                                    st.dataframe(dfs["personnel"], use_container_width=True)
                            # Optionally extract company insights
                            insights_df = pd.DataFrame()
                            if extract_insights:
                                with st.spinner("Extracting company insights (this may take a while)..."):
                                    insights_automation_id = "63f878da7022e05c1192df29"  # LinkedIn Company Insights
                                    insights_result = self.linkedin_api.run_automation(
                                        name="Company Insights",
                                        description="Extract LinkedIn company insights",
                                        automation_id=insights_automation_id,
                                        connected_account_id=connected_account_id,
                                        timezone="Asia/Kolkata",
                                        inputs={"liCompanyUrl": company_url}
                                    )
                                    insights_data = insights_result.get("data", {})
                                    insights_execution_id = insights_data.get("id") or insights_data.get("workflowId")
                                    insights_final_result = None
                                    if insights_execution_id:
                                        for _ in range(600):  # wait up to 10 minutes
                                            insights_final_result = self.linkedin_api.get_execution_result(insights_execution_id)
                                            if insights_final_result.get("data"):
                                                break
                                            time.sleep(1)
                                    if insights_final_result and "data" in insights_final_result:
                                        insights_data = insights_final_result["data"]
                                        if isinstance(insights_data, list):
                                            insights_df = pd.json_normalize(insights_data)
                                        elif isinstance(insights_data, dict):
                                            insights_df = pd.json_normalize([insights_data])
                                        insights_df = self.remove_empty_columns(insights_df)
                                        if not insights_df.empty:
                                            st.subheader(f"Company Insights ({len(insights_df)})")
                                            st.dataframe(insights_df, use_container_width=True)
                                            dfs["company_insights"] = insights_df
                            # Export button
                            if dfs:
                                excel_buffer = io.BytesIO()
                                with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                                    for sheet_name, df in dfs.items():
                                        if not df.empty:
                                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                                excel_buffer.seek(0)
                                if st.download_button(
                                    label="Download as Excel",
                                    data=excel_buffer.getvalue(),
                                    file_name=f"linkedin_company_data.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                ):
                                    filepath = self.data_processor.export_to_excel(dfs, "company")
                                    st.markdown(f"<p class='success-msg'>File saved to: {filepath}</p>", unsafe_allow_html=True)
                            else:
                                st.warning("No data found for this company URL.")
                    except Exception as e:
                        app_logger.error("Error in company extraction: {}", str(e))
                        st.error(f"An error occurred: {str(e)}")
                else:
                    st.warning("Please enter a valid LinkedIn company URL.")
            if company_url or st.session_state.get('company_extraction_result'):
                st.markdown("</div>", unsafe_allow_html=True)

    def decision_maker_pipeline_page(self):
        """Automated pipeline: Keyword Search → Author/Liker/Commenter Profile Extraction → Merge → Filter → Export"""
        st.markdown("<div class='section-header'>Decision-Maker Pipeline (Fully Automated)</div>", unsafe_allow_html=True)
        st.caption("Enter a keyword and run the full LinkedIn workflow: search posts, extract authors, likers, commenters, merge, filter for decision-makers, and export.")
        keyword = st.text_input("Enter keyword(s) for LinkedIn post search", key="pipeline_keyword_auto")
        search_limit = st.number_input("Number of posts to extract", min_value=1, max_value=100, value=10, step=1, key="pipeline_search_limit_auto")
        run_pipeline = st.button("Run Pipeline", key="pipeline_run_all")
        if run_pipeline and keyword:
            with st.spinner("Running full pipeline. This may take several minutes..."):
                # Step 1: Keyword Search
                automation_id = "64099c6e0936e46db5d76f4c"
                connected_account_id = self._get_automation_and_account("keyword search")[1]
                api_inputs = {"liPostSearchUrl": keyword, "maxCountPostSearch": int(search_limit)}
                result = self.linkedin_api.run_automation(
                    name="Pipeline Keyword Search",
                    description="Pipeline: Search LinkedIn posts by keywords",
                    automation_id=automation_id,
                    connected_account_id=connected_account_id,
                    timezone="Asia/Kolkata",
                    inputs=api_inputs
                )
                data = result.get("data", {})
                execution_id = data.get("id") or data.get("workflowId")
                final_result = None
                if execution_id:
                    for _ in range(120):
                        final_result = self.linkedin_api.get_execution_result(execution_id)
                        if final_result.get("data"):
                            break
                        time.sleep(1)
                if not (final_result and "data" in final_result):
                    st.error("No posts found for the given keyword.")
                    return
                posts_df = pd.json_normalize(final_result["data"])
                posts_df = self.remove_empty_columns(posts_df)

                # Automatically download post data after extraction
                post_data_path = os.path.join(os.path.expanduser("~"), "Downloads", "extracted_posts.xlsx")
                with pd.ExcelWriter(post_data_path, engine="openpyxl") as writer:
                    posts_df.to_excel(writer, sheet_name="posts", index=False)
                st.success(f"Post data automatically downloaded to {post_data_path}")

                # Step 2: Extract Author Profiles
                col_candidates = [c for c in posts_df.columns if c.lower() == "lipublicprofileurl"]
                if not col_candidates:
                    st.error("No 'liPublicProfileURL' column found in the posts data. Cannot proceed.")
                    return
                col = col_candidates[0]
                author_urls = posts_df[col].dropna().unique().tolist()
                author_profiles = []
                for url in author_urls:
                    result = self.linkedin_api.run_automation(
                        name="Pipeline Author Profile Extraction",
                        description="Pipeline: Extract author profile",
                        automation_id="63f48ee97022e05c116fc798",
                        connected_account_id=self._get_automation_and_account("profile extraction")[1],
                        timezone="Asia/Kolkata",
                        inputs={"liProfileUrl": url}
                    )
                    data = result.get("data", {})
                    execution_id = data.get("id") or data.get("workflowId")
                    final_result = None
                    if execution_id:
                        for _ in range(60):
                            final_result = self.linkedin_api.get_execution_result(execution_id)
                            if final_result.get("data"):
                                break
                            time.sleep(1)
                    if final_result and "data" in final_result:
                        author_profiles.append(final_result["data"])
                author_profiles_df = self.expand_profiles_to_df(author_profiles)
                author_profiles_df = self.remove_empty_columns(author_profiles_df)
                # Step 3: Extract Likers
                post_urls = posts_df['postUrl'].dropna().unique().tolist() if 'postUrl' in posts_df else []
                all_likers = []
                for url in post_urls:
                    result = self.linkedin_api.run_automation(
                        name="Pipeline Post Likers Extraction",
                        description="Pipeline: Extract post likers",
                        automation_id="63fc575f7022e05c11bba145",
                        connected_account_id=self._get_automation_and_account("post likers extractor")[1],
                        timezone="Asia/Kolkata",
                        inputs={"liPostUrl": url}
                    )
                    data = result.get("data", {})
                    execution_id = data.get("id") or data.get("workflowId")
                    final_result = None
                    if execution_id:
                        for _ in range(60):
                            final_result = self.linkedin_api.get_execution_result(execution_id)
                            if final_result.get("data"):
                                break
                            time.sleep(1)
                    if final_result and "data" in final_result:
                        likers = final_result["data"]
                        if isinstance(likers, list):
                            all_likers.extend(likers)
                        elif isinstance(likers, dict):
                            all_likers.append(likers)
                likers_df = pd.json_normalize(all_likers)
                likers_df = self.remove_empty_columns(likers_df)
                # Step 4: Extract Commenters
                all_commenters = []
                for url in post_urls:
                    result = self.linkedin_api.run_automation(
                        name="Pipeline Post Commenters Extraction",
                        description="Pipeline: Extract post commenters",
                        automation_id="63fc8cd27022e05c113c3c73",
                        connected_account_id=self._get_automation_and_account("post commenters extractor")[1],
                        timezone="Asia/Kolkata",
                        inputs={"liPostUrl": url}
                    )
                    data = result.get("data", {})
                    execution_id = data.get("id") or data.get("workflowId")
                    final_result = None
                    if execution_id:
                        for _ in range(60):
                            final_result = self.linkedin_api.get_execution_result(execution_id)
                            if final_result.get("data"):
                                break
                            time.sleep(1)
                    if final_result and "data" in final_result:
                        commenters = final_result["data"]
                        if isinstance(commenters, list):
                            all_commenters.extend(commenters)
                        elif isinstance(commenters, dict):
                            all_commenters.append(commenters)
                commenters_df = pd.json_normalize(all_commenters)
                commenters_df = self.remove_empty_columns(commenters_df)
                # Step 5: Liker Profile Scraping
                liker_urls = likers_df['profileUrl'].dropna().unique().tolist() if 'profileUrl' in likers_df else []
                liker_profiles = []
                for url in liker_urls:
                    result = self.linkedin_api.run_automation(
                        name="Pipeline Liker Profile Extraction",
                        description="Pipeline: Extract liker profile",
                        automation_id="63f48ee97022e05c116fc798",
                        connected_account_id=self._get_automation_and_account("profile extraction")[1],
                        timezone="Asia/Kolkata",
                        inputs={"liProfileUrl": url}
                    )
                    data = result.get("data", {})
                    execution_id = data.get("id") or data.get("workflowId")
                    final_result = None
                    if execution_id:
                        for _ in range(60):
                            final_result = self.linkedin_api.get_execution_result(execution_id)
                            if final_result.get("data"):
                                break
                            time.sleep(1)
                    if final_result and "data" in final_result:
                        liker_profiles.append(final_result["data"])
                liker_profiles_df = self.expand_profiles_to_df(liker_profiles)
                liker_profiles_df = self.remove_empty_columns(liker_profiles_df)
                # Step 6: Commenter Profile Scraping
                commenter_urls = commenters_df['profileUrl'].dropna().unique().tolist() if 'profileUrl' in commenters_df else []
                commenter_profiles = []
                for url in commenter_urls:
                    result = self.linkedin_api.run_automation(
                        name="Pipeline Commenter Profile Extraction",
                        description="Pipeline: Extract commenter profile",
                        automation_id="63f48ee97022e05c116fc798",
                        connected_account_id=self._get_automation_and_account("profile extraction")[1],
                        timezone="Asia/Kolkata",
                        inputs={"liProfileUrl": url}
                    )
                    data = result.get("data", {})
                    execution_id = data.get("id") or data.get("workflowId")
                    final_result = None
                    if execution_id:
                        for _ in range(60):
                            final_result = self.linkedin_api.get_execution_result(execution_id)
                            if final_result.get("data"):
                                break
                            time.sleep(1)
                    if final_result and "data" in final_result:
                        commenter_profiles.append(final_result["data"])
                commenter_profiles_df = self.expand_profiles_to_df(commenter_profiles)
                commenter_profiles_df = self.remove_empty_columns(commenter_profiles_df)
                # Step 7: Merge All Profile Data
                merged_df = pd.concat([author_profiles_df, liker_profiles_df, commenter_profiles_df], ignore_index=True)
                merged_df = merged_df.drop_duplicates(subset=[c for c in merged_df.columns if 'linkedin' in c.lower() or 'profileurl' in c.lower() or 'url' in c.lower()])

                # Save raw data (before filtering) to session state
                raw_data_buffer = io.BytesIO()
                with pd.ExcelWriter(raw_data_buffer, engine="openpyxl") as writer:
                    merged_df.to_excel(writer, sheet_name="raw_data", index=False)
                raw_data_buffer.seek(0)
                st.session_state["raw_data_buffer"] = raw_data_buffer.getvalue()
                st.session_state["raw_data_filename"] = "raw_data.xlsx"

                # Step 8: Filter Decision-Makers
                keywords = ["founder", "cxo", "ceo", "coo","cio","cto","chro","cpo","cro","CISO","clo","cmo","director", "vp", "head", "decision", "leader", "Manager", "executive", "owner", "president", "Co-Founder", "Chief", "Head of", "Lead"]
                # Ensure all columns in str_cols are converted to strings before applying .str operations
                str_cols = merged_df.select_dtypes(include=["object", "string"]).columns
                merged_df[str_cols] = merged_df[str_cols].applymap(lambda x: str(x) if not pd.isna(x) else '')
                combined = merged_df[str_cols].apply(lambda x: ' '.join([str(i) if not pd.isna(i) else '' for i in x]), axis=1)
                combined = combined.astype(str).str.lower()
                mask = combined.apply(lambda x: any(k in x for k in keywords))
                filtered_df = merged_df[mask]
                # Define important columns in the specified order
                important_columns = [
                    "liPublicProfileUrl", "firstName", "lastName", "companyName", "liCompanyPublicUrl", "headcountRange", "jobLocationArea", "jobTitle", "jobTenure",
                    "profileDescription", "headline", "emailAddressPersonal", "profileLocationCountry", "profileLocationCity", "profileLocationArea", "locationCountryCode", "industry"
                ]
                # Filter to only important columns that exist in the DataFrame
                display_df = filtered_df[[col for col in important_columns if col in filtered_df.columns]]
                st.success(f"Pipeline complete! Filtered {len(filtered_df)} high-level profiles.")
                st.dataframe(display_df, use_container_width=True)

                # Automatically download final decision-maker data
                final_data_path = os.path.join(os.path.expanduser("~"), "Downloads", "decision_makers.xlsx")
                with pd.ExcelWriter(final_data_path, engine="openpyxl") as writer:
                    filtered_df.to_excel(writer, sheet_name="decision_makers", index=False)
                st.success(f"Decision-maker data automatically downloaded to {final_data_path}")

                # Export full filtered data
                excel_buffer_full = io.BytesIO()
                with pd.ExcelWriter(excel_buffer_full, engine="openpyxl") as writer:
                    filtered_df.to_excel(writer, sheet_name="decision_makers", index=False)
                excel_buffer_full.seek(0)
                # Export only important columns
                excel_buffer_important = io.BytesIO()
                with pd.ExcelWriter(excel_buffer_important, engine="openpyxl") as writer:
                    display_df.to_excel(writer, sheet_name="important_columns", index=False)
                excel_buffer_important.seek(0)
                # Store in session_state to persist until navigation or new search
                st.session_state["decision_makers_full_buffer"] = excel_buffer_full.getvalue()
                st.session_state["decision_makers_important_buffer"] = excel_buffer_important.getvalue()
                st.session_state["decision_makers_full_filename"] = "decision_makers_full.xlsx"
                st.session_state["decision_makers_important_filename"] = "decision_makers_important_columns.xlsx"

            # Provide a single 'Raw Data Download' button
            if "raw_data_buffer" in st.session_state:
                st.download_button(
                    label="Download Raw Data",
                    data=st.session_state["raw_data_buffer"],
                    file_name=st.session_state["raw_data_filename"],
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    def run(self):
        """Run the Streamlit application"""
        self.setup_page()
        selected_page = self.display_navigation()
        if selected_page == "Keyword Search":
            self.keyword_search_page()
        elif selected_page == "Post Extraction":
            self.post_extraction_page()
        elif selected_page == "Profile Extraction":
            self.profile_extraction_page()
        elif selected_page == "Company Extraction":
            self.company_extraction_page()
        elif selected_page == "Decision-Maker Pipeline":
            self.decision_maker_pipeline_page()
        elif selected_page == "LinkedIn Comment Generator":
            linkedin_comment_page()
        elif selected_page == "AI Comment Generator (Blended)":
            from app import main as blended_comment_ui
            blended_comment_ui()

if __name__ == "__main__":
    app = LinkedInExtractorApp()
    app.run()