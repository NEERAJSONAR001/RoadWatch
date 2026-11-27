@echo off
title RoadWatch Demo App
color 0A
echo.
echo  =====================================================
echo   RoadWatch - Pothole Detection Demo
echo   MSc Computer Science - KC College, Mumbai
echo   Neeraj Sonar - Roll: 26KFMSCCS26
echo  =====================================================
echo.
echo  Starting demo... browser will open automatically.
echo  Press Ctrl+C to stop the app.
echo.

cd /d "%~dp0"

C:\Users\admin\AppData\Local\Programs\Python\Python314\python.exe -m streamlit run app.py --server.port 8501 --browser.gatherUsageStats false --theme.base dark

pause
