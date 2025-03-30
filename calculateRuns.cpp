#include <string>
#include <iostream>
#include <cmath>


int main(int argc, char *argv[])
{
    float confidence;
    float error;
    int threads = 1024;
    if (argc < 3)
    {
        confidence = 0.95f;
        error = 0.005f;
        
    }
    else{
        confidence = std::stof(argv[1]);
        error = std::stof(argv[2]);
    }
    std::cout << "confidence: "  << confidence << " error: " << error << std::endl;
    double number =  ceil((log(2/(1-confidence)))/(2*error*error));
    std::cout << "number of executions: " << number << std::endl;
    float executionCount = ceil(number/32);
    std::cout << "number of executions: " << executionCount << std::endl;
    std::cout << "number of executions: " << executionCount*32 << std::endl;
    return 0;
}