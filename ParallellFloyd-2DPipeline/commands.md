## Run

mpiexec -n 4 "C:\Users\caleb\source\repos\ParallellFloyd-2DPipeline\x64\Debug\ParallellFloyd-2DPipeline.exe
& "~\source\repos\ParallellFloyd-2DPipeline\x64\Debug\ParallellFloyd-2DPipeline.exe" 1

## Generate input graph

& "~\source\repos\ParallellFloyd-2DPipeline\x64\Debug\GraphGenerator.exe" 9

## Visualize Input

python .\samples\visualization.py "1"

## Visualize Output

python .\ParallellFloyd-2DPipeline\visualization.py "C:\Users\caleb\source\repos\ParallellFloyd-2DPipeline\ParallellFloyd-2DPipeline\output_matrix.txt"