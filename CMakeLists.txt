# Necessita de versão >= 3.10
cmake_minimum_required(VERSION 3.10)
project(HighPrecisionProtonTherapy VERSION 1.0)

# Setar a norma de C++ a usar
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Definir warnings e otimizações
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -ffast-math")

# Adicionar o include directory
include_directories(src/mc_engine)

# Ficheiros fonte
set(SOURCES
    src/mc_engine/main.cpp
    src/mc_engine/geometry.cpp
    src/mc_engine/physics.cpp
    src/mc_engine/transport.cpp
    src/mc_engine/export.cpp
)

# Executável
add_executable(mc_engine ${SOURCES})
