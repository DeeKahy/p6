
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <functional>
#include <random>
#include <stdio.h>
#include <iostream>
#include <vector>
struct Transition {
	int type;
	int from;
	int to;
	float guard[2];
	std::function<bool(float*)> functionToRun;
};
struct Euler
{
	int timesCorrect = 0;
	int places[2]{ 0 };
	float accumulation = 0; 
	std::vector<Transition> transitions;

};



int main()
{
	Euler euler;

	Transition timeIncrease;
	timeIncrease.from = 0;
	timeIncrease.to = 0;
	timeIncrease.guard[0] = 0;
	timeIncrease.guard[1] = std::numeric_limits<float>::max();
	timeIncrease.functionToRun = [](float* value) {

		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> dis(0.0, 1.0);
		float randomValue = dis(gen);
		*value += randomValue;
		return false;
		};
	euler.transitions.push_back(timeIncrease);
	Transition timeOut;
	timeOut.from = 0;
	timeOut.to = 1;
	timeOut.guard[0] = 1;
	timeOut.guard[1] = std::numeric_limits<float>::max();

	timeOut.functionToRun = [](float* value) {
		return true;

		};
	euler.transitions.push_back(timeOut);
	const float EPSILON = 0.001f;
	for (size_t i = 0; i < 1000000; i++)
	{
		int timesFired = 0;
		float test = 0;
		bool shouldBreak = false;
		while (!shouldBreak)
		{
			Transition youngest = euler.transitions[0];
			for (size_t j = 0; j < euler.transitions.size(); j++)
			{
				if (test >= euler.transitions[j].guard[0] && test <= euler.transitions[j].guard[1] && youngest.guard[0] < euler.transitions[j].guard[0])
				{
					youngest = euler.transitions[j];
				}

			}
			shouldBreak = youngest.functionToRun(&test);
			timesFired += 1;
			if (std::abs(test - 1.0f) < EPSILON)
			{
				euler.timesCorrect += 1;
				euler.accumulation += timesFired;

			}
		}

	}
	std::cout << euler.accumulation  /  euler.timesCorrect ;
	return 0;
}
