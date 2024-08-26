## Run

mpiexec -n 4 "C:\Users\caleb\source\repos\ParallellFloyd-2DPipeline\x64\Debug\ParallellFloyd-2DPipeline.exe
& "~\source\repos\ParallellFloyd-2DPipeline\x64\Debug\ParallellFloyd-2DPipeline.exe" samples\1

mpiexec -n 4 "C:\Users\caleb\source\repos\ParallellFloyd-2DPipeline\x64\Debug\ParallellFloyd-2DPipeline.exe" .\samples\3\matrix.txt

## Generate input graph

& "~\source\repos\ParallellFloyd-2DPipeline\x64\Debug\GraphGenerator.exe" 4

## Generate visualizations
python .\samples\visualization.py "1"

