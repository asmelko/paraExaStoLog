#pragma once

#include <chrono>

class Timer
{
public:
	Timer() {}

	void Start() { start_time_ = std::chrono::system_clock::now(); }

	void Stop() { end_time_ = std::chrono::system_clock::now(); }

	double Seconds() const { return std::chrono::duration_cast<std::chrono::seconds>(end_time_ - start_time_).count(); }

	double Millisecs() const
	{
		return std::chrono::duration_cast<std::chrono::milliseconds>(end_time_ - start_time_).count();
	}

	double Microsecs() const
	{
		return std::chrono::duration_cast<std::chrono::microseconds>(end_time_ - start_time_).count();
	}

private:
	std::chrono::time_point<std::chrono::system_clock> start_time_;
	std::chrono::time_point<std::chrono::system_clock> end_time_;
};

// Times op's execution using the timer t
#define TIME_OP(t, op)                                                                                                 \
	{                                                                                                                  \
		t.Start();                                                                                                     \
		(op);                                                                                                          \
		t.Stop();                                                                                                      \
	}
