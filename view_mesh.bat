@echo off
title Open3D Interactive Mesh Viewer
cd /d "%~dp0"
echo ========================================================
echo Launching Open3D Interactive 3D Mesh Viewer...
echo Loading: runs/drone3d_custom/stage_07_mesh/mesh_raw.ply
echo ========================================================
python tools\view_mesh.py runs\drone3d_custom\stage_07_mesh\mesh_raw.ply
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Trying fallback OBJ format...
    python tools\view_mesh.py runs\drone3d_custom\stage_07_mesh\mesh_raw.obj
)
pause
