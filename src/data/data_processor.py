import pandas as pd
import os
import json
from datetime import datetime
from ..logger import app_logger

class DataProcessor:
    """Class for processing and exporting LinkedIn data"""
    
    @staticmethod
    def convert_to_dataframe(data, data_type):
        """Convert API response data to pandas DataFrame
        
        Args:
            data: API response data
            data_type: Type of data (posts, profile, company, etc.)
            
        Returns:
            Pandas DataFrame
        """
        app_logger.info("Converting {} data to DataFrame", data_type)
        
        try:
            if data_type == "keyword_posts" or data_type == "recent_posts":
                # Process list of posts
                df = pd.json_normalize(data.get("data", []))
            elif data_type == "post":
                # Process single post data
                post_data = data.get("data", {})
                
                # Create main post dataframe
                post_df = pd.json_normalize(post_data)
                
                # Process reactors and commenters if available
                reactors_df = pd.DataFrame()
                if "reactors" in post_data:
                    reactors_df = pd.json_normalize(post_data["reactors"])
                
                commenters_df = pd.DataFrame()
                if "comments" in post_data:
                    commenters_df = pd.json_normalize(post_data["comments"])
                
                # Return dictionary of dataframes
                return {
                    "post": post_df,
                    "reactors": reactors_df,
                    "commenters": commenters_df
                }
            elif data_type == "profile":
                # Process profile data
                df = pd.json_normalize(data.get("data", {}))
            elif data_type == "company":
                # Process company data
                company_data = data.get("data", {})
                
                # Create main company dataframe
                company_df = pd.json_normalize(company_data)
                
                # Process key personnel if available
                personnel_df = pd.DataFrame()
                if "key_personnel" in company_data:
                    personnel_df = pd.json_normalize(company_data["key_personnel"])
                
                # Return dictionary of dataframes
                return {
                    "company": company_df,
                    "personnel": personnel_df
                }
            else:
                app_logger.error("Unknown data type: {}", data_type)
                raise ValueError(f"Unknown data type: {data_type}")
                
            return df
            
        except Exception as e:
            app_logger.error("Error converting data to DataFrame: {}", str(e))
            raise
    
    @staticmethod
    def export_to_excel(data, data_type):
        """Export data to Excel file
        
        Args:
            data: Data to export (DataFrame or dict of DataFrames)
            data_type: Type of data (posts, profile, company, etc.)
            
        Returns:
            Path to the saved Excel file
        """
        app_logger.info("Exporting {} data to Excel", data_type)
        
        try:
            # Create downloads directory if it doesn't exist
            downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
            if not os.path.exists(downloads_dir):
                os.makedirs(downloads_dir)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"linkedin_{data_type}_{timestamp}.xlsx"
            filepath = os.path.join(downloads_dir, filename)
            
            # Export to Excel
            with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
                if isinstance(data, dict):
                    # Multiple sheets for different data types
                    for sheet_name, df in data.items():
                        if not df.empty:
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                else:
                    # Single sheet for simple data
                    data.to_excel(writer, sheet_name=data_type, index=False)
            
            app_logger.info("Data exported to {}", filepath)
            return filepath
            
        except Exception as e:
            app_logger.error("Error exporting data to Excel: {}", str(e))
            raise