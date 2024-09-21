## Build

docker build -f "./Floyd-WarshallMPI/Dockerfile" -t floyd-warshall-mpi:latest "./"

## Run

docker run -it --rm -t floyd-warshall-mpi:latest

## Push

docker tag local-image:tagname patautista/floyd-warshall-mpi:latest
docker push patautista/floyd-warshall-mpi:latest

