#include "parser.h"

__global__ void run(float *results, Tapn *templateNet, int placesCount)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Correct bounds check

    // Initialize TAPN
    Tapn myNet;
    myNet.currentTime = 0.0f;
    myNet.steps = 0;
    myNet.placesCount = placesCount;
    myNet.transitionsCount = templateNet->transitionsCount;
    Transition transitions[20];
    myNet.transitions = transitions;

    for (int i = 0; i < myNet.transitionsCount; i++)
    {
        // Copy the entire struct at once
        memcpy(&myNet.transitions[i], &templateNet->transitions[i], sizeof(Transition));
    }
    Place myPlaces[20];


    float token = 0.0f;
    float tokens[1]{token};
    myPlaces[0].addTokens(tokens, 1);

    // Run simulation
    myNet.run(myPlaces);

    // Direct assignment instead of accumulation
    results[tid] += myNet.steps; // Using = instead of +=

    // Clean up
    delete[] myPlaces;
    delete[] myNet.transitions;
}

rapidxml::xml_node<> *loadPNML(std::string file)
{
    rapidxml::file<> xmlFile(file.c_str());
    rapidxml::xml_document<> doc;
    doc.parse<0>(xmlFile.data());
    return doc.first_node("pnml");
}

// Function to parse arc inscriptions
std::tuple<float, float, int> parseInscription(const std::string &inscription)
{
    // Default values
    float lower = 0.0f;
    float upper = FLT_MAX; // Default to [0,inf)
    int weight = 1;

    // Check if the inscription is just a number (weight only)
    if (inscription.find('[') == std::string::npos)
    {
        try
        {
            // If it's just a number, treat it as the weight with default interval
            weight = std::stoi(inscription);
            return std::make_tuple(lower, upper, weight);
        }
        catch (const std::exception &e)
        {
            throw std::invalid_argument("Invalid weight format: " + inscription);
        }
    }

    // Find the colon that separates interval and weight
    size_t colonPos = inscription.find(':');

    // Extract interval part
    std::string intervalPart;

    if (colonPos != std::string::npos)
    {
        // Format is "[lower,upper):weight"
        intervalPart = inscription.substr(0, colonPos);
        weight = std::stoi(inscription.substr(colonPos + 1));
    }
    else
    {
        // Format is just "[lower,upper)"
        intervalPart = inscription;
    }

    // Now parse the interval
    // Remove brackets
    std::string content = intervalPart.substr(1, intervalPart.length() - 2);
    size_t commaPos = content.find(',');

    if (commaPos == std::string::npos)
    {
        throw std::invalid_argument("Invalid interval format: " + intervalPart);
    }

    std::string lowerBound = content.substr(0, commaPos);
    std::string upperBound = content.substr(commaPos + 1);

    lower = std::stof(lowerBound);

    // Handle infinity
    if (upperBound == "inf")
    {
        upper = FLT_MAX;
    }
    else
    {
        upper = std::stof(upperBound);
    }

    return std::make_tuple(lower, upper, weight);
}

int main()
{
    auto start = std::chrono::high_resolution_clock::now();
    rapidxml::xml_node<> *test = loadPNML(path);
    rapidxml::xml_node<> *net = test->first_node("net");

    std::map<std::string, Place *> places;
    rapidxml::xml_node<> *place = net->first_node("place");
    while (place)
    {
        Place *newPlace = new Place;
        std::string newString = place->first_attribute("id")->value();
        places.insert(std::make_pair(newString, newPlace));

        place = place->next_sibling("place");
    }
    std::map<std::string, Transition *> transitions;
    rapidxml::xml_node<> *transition = net->first_node("transition");
    while (transition)
    {
        Transition *newTransition = new Transition;
        newTransition->urgent = transition->first_attribute("urgent")->value() == "true";
        std::string distribution = transition->first_attribute("distribution")->value();
        if (distribution == "uniform")
        {
            newTransition->distribution.type = UNIFORM;
            newTransition->distribution.a = std::stof(transition->first_attribute("a")->value());
            newTransition->distribution.b = std::stof(transition->first_attribute("b")->value());
        }
        else if (distribution == "constant")
        {
            newTransition->distribution.type = CONSTANT;
            newTransition->distribution.a = std::stof(transition->first_attribute("value")->value());
        }
        else if (distribution == "normal")
        {
            newTransition->distribution.type = NORMAL;
        }
        else if (distribution == "exponential")
        {
            newTransition->distribution.type = EXPONENTIAL;
        }

        std::string newString = transition->first_attribute("id")->value();
        transitions.insert(std::make_pair(newString, newTransition));

        transition = transition->next_sibling("transition");
    }
    std::map<std::string, Arc *> arcs;
    rapidxml::xml_node<> *arc = net->first_node("arc");
    while (arc)
    {
        std::string newString = arc->first_attribute("id")->value();
        std::string source = arc->first_attribute("source")->value();
        std::string target = arc->first_attribute("target")->value();
        std::string inscription = arc->first_attribute("inscription")->value();
        std::string type = arc->first_attribute("type")->value();

        int index{0};
        if (transitions.find(source) != transitions.end())
        {

            for (auto it = transitions.begin(); it != transitions.end(); ++it, ++index)
            {
                if (it->first == source)
                {
                    int index2{0};

                    for (auto place = places.begin(); place != places.end(); ++place, ++index2)
                    {
                        ;
                        if (place->first == target)
                        {
                            Transition *tran = transitions.find(source)->second;
                            tran->outputArcs[tran->outputArcsCount].output = index2;
                            tran->outputArcs[tran->outputArcsCount].isTransport = type == "transport";
                            tran->outputArcsCount++;
                            break;
                        }
                    }
                }
            }
        }
        else
        {
            for (auto it = transitions.begin(); it != transitions.end(); ++it, ++index)
            {
                if (it->first == target)
                {
                    int index2{0};

                    for (auto place = places.begin(); place != places.end(); ++place, ++index2)
                    {
                        ;
                        if (place->first == source)
                        {
                            Transition *tran = transitions.find(target)->second;
                            auto [lower, upper, weight] = parseInscription(arc->first_attribute("inscription")->value());
                            if (type == "transport")
                            {
                                tran->inputArcs[tran->inputArcsCount].type = TRANSPORT;
                            }
                            else if (type == "timed")
                            {
                                tran->inputArcs[tran->inputArcsCount].type = INPUT;
                            }
                            else if (type == "inhibitor")
                            {
                                tran->inputArcs[tran->inputArcsCount].type = INHIBITOR;
                            }
                            tran->inputArcs[tran->inputArcsCount].timings[0] = lower;
                            tran->inputArcs[tran->inputArcsCount].timings[1] = upper;
                            tran->inputArcs[tran->inputArcsCount].weight = weight;
                            tran->inputArcs[tran->inputArcsCount].place = index2;
                            tran->inputArcs[tran->inputArcsCount].constraint = 0;
                            tran->inputArcsCount++;

                            break;
                        }
                    }
                    break;
                }
            }
        }
        arc = arc->next_sibling("arc");
    }
    // Add this after transitions are populated and before CUDA memory allocation

    Place *placeArray = new Place[places.size()];
    int index = 0;
    for (const auto &pair : places)
    {
        placeArray[index++] = *pair.second;
    }
    Transition *transitionArray = new Transition[transitions.size()];
    index = 0;
    for (const auto &pair : transitions)
    {
        transitionArray[index++] = *pair.second;
    }

    Tapn tapn;
    tapn.transitions = transitionArray;
    tapn.placesCount = places.size();
    tapn.transitions = transitionArray;
    tapn.transitionsCount = transitions.size();

    float confidence;
    float error;
    unsigned long long threads = 1024;
    unsigned long long blockCount = 2048;

    confidence = 0.95f;
    error = 0.0005f;

    std::cout << "confidence: " << confidence << " error: " << error << "\n";
    float number = ceil((log(2 / (1 - confidence))) / (2 * error * error));
    std::cout << "execution calculated: " << number << "\n";
    unsigned long long loopCount = ceil(number / (blockCount * threads));
    std::cout << "loop count: " << loopCount << "\n";
    unsigned long long N{blockCount * threads};
    std::cout << "number of executions run: " << N * loopCount << "\n";

    float *d_results;
    Place *d_places;
    cudaMalloc((void **)&d_results, N * sizeof(float));
    cudaMemset(d_results, 0, N * sizeof(float));
    // Allocate and copy places to device
    cudaMalloc((void **)&d_places, places.size() * sizeof(Place));
    cudaMemcpy(d_places, placeArray, places.size() * sizeof(Place), cudaMemcpyHostToDevice);
    Transition *d_transitions;
    // Allocate and copy transitions to device
    cudaMalloc((void **)&d_transitions, transitions.size() * sizeof(Transition));
    cudaMemcpy(d_transitions, transitionArray, transitions.size() * sizeof(Transition), cudaMemcpyHostToDevice);

    // Create a template TAPN
    Tapn *d_tapn;
    cudaMalloc((void **)&d_tapn, sizeof(Tapn));
    Tapn h_tapn;
    h_tapn.placesCount = places.size();
    h_tapn.transitionsCount = transitions.size();
    h_tapn.transitions = d_transitions;
    cudaMemcpy(d_tapn, &h_tapn, sizeof(Tapn), cudaMemcpyHostToDevice);

    // Launch kernel with deep copying
    for (size_t i = 0; i < loopCount; i++)
    {
        run<<<blockCount, threads>>>(d_results, d_tapn, places.size());
        cudaDeviceSynchronize();
    }

    cudaError_t errSync = cudaDeviceSynchronize();
    cudaError_t errAsync = cudaGetLastError();

    if (errSync != cudaSuccess)
    {
        printf("Sync error: %s\n", cudaGetErrorString(errSync));
    }
    if (errAsync != cudaSuccess)
    {
        printf("Launch error: %s\n", cudaGetErrorString(errAsync));
    }
    thrust::device_ptr<float> d_ptr = thrust::device_pointer_cast(d_results);
    double tot = thrust::reduce(d_ptr, d_ptr + N);
    std::cout << "Success rate: " << tot / (N * loopCount) << "\n";

    // Clean up
    cudaFree(d_places);
    cudaFree(d_transitions);
    cudaFree(d_tapn);
    cudaFree(d_results);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "time run: " << duration.count() << "\n";
}