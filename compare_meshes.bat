@echo off
title 3D Mesh Comparator - 35m vs 40m
cd /d "%~dp0"
echo =======================================================================
echo Launching Interactive 3D Mesh Comparison Tool (Open3D)...
echo.
echo Hotkeys inside the viewer:
echo   [1] 35m Mesh (Core Terrain, Roads, Buildings - Solid Single Surface)
echo   [2] 40m Mesh (Extended Ground Margin)
echo   [3] Stage 07 Mesh (Original with Horizon Spire for Reference)
echo.
echo Camera viewpoint, angle, and zoom remain 100%% identical across switches!
echo =======================================================================
python tools\compare_meshes.py
pause
