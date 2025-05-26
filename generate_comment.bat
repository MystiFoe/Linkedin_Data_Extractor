@echo off
echo AI LinkedIn Comment Generator
echo.
set /p input="Enter LinkedIn post or URL: "
python ai_comment_generator.py --post "%input%"
echo.
pause