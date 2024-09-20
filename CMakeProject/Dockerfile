# Dockerfile

# Use an official Ubuntu base image
FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    openmpi-bin \
    libopenmpi-dev

# Create a non-root user
RUN useradd -ms /bin/bash mpiuser
USER mpiuser

# Set the working directory inside the container
WORKDIR /usr/src/app/

# Copy the entire project into the container
COPY --chown=mpiuser:mpiuser CMakeProject .

# Create the build directory and run CMake
RUN mkdir build && cd build \
    && cmake .. \
    && make

WORKDIR /usr/src/app/build

# Set the entry point to launch a shell
CMD ["/bin/bash"]
