
cmake_minimum_required(VERSION 3.10)
project(PNMLParser)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find pugixml package
find_package(pugixml REQUIRED)

# Add include directories
include_directories(include)

# Add source files
set(SOURCES
    src/pnml_parser.cpp
    src/petri_net.cpp
    src/arc.cpp
    src/main.cpp
)

# Create executable
add_executable(pnml_parser ${SOURCES})
target_link_libraries(pnml_parser pugixml)