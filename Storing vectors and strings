//Assignment 3 OOP 2020
// Simple code to store courses using vectors and strings


/*
Assignment 3
physics course database
Please note: These are the only requirements; expected coding time 2.5 hours.
Write a C++ program to store and print physics courses
Allow the user to enter an arbitrary number of courses
Ask the user to provide a course code (as an integer, e.g. 30762) and title (as a string, e.g. Object-Oriented Programming in C++), both on a single line
Use a string stream to create a string containing the full course title, e.g.
PHYS 30762 Object-Oriented Programming in C++
Each of these strings should be stored in a vector
Print out full course list using an iterator


A file with sample inputs is available https://theory.physics.manchester.ac.uk/~mccsnrw/cplusplus/skeleton/ass3/courselist.dat

Assignment 3
physics course database
Your program should correctly use
a vector of strings to store course data (0.5 mark)
a string stream to combine the integer course code and title (0.5 mark)
an iterator to print out information for each course (1 mark)
Your code should also be able to print out a list of courses for a particular year, as identified by the first digit of the course code (1 mark)
It should be possible to sort the list of courses by title or course code (1 mark)
Negative marks:
NEW: If program does not compile without errors (-1 mark)
Does not adhere to house style (-1 mark)
Not present for marking (-1 mark)
*/

#include<iostream>
#include<string>
#include<sstream>
#include<vector>
#include <algorithm> //sort function


int main()
{	std::string sort_option;
	const std::string degreeCode("PHYS");


	int input;
	int course_number;
	std::string course_name;
	std::vector<std::string> course_inputs;
	std::vector<std::string> course_specific_year;

	int year_choice;

	// Gather list of courses and their codes from user,
	// storing data as a vector of strings
	bool not_finished{ true };
	do
	{
		std::cout << "Please enter a course name beginning with course number and then title. \n (or x to finish): \n";

		std::cin >> course_number;
		 //need to tell cin not to skip spaces, takes whole line input

		if (std::cin.fail()) {
			not_finished = false;
			std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		}
		else {

			std::getline(std::cin, course_name);
			std::ostringstream output_stream;
			output_stream << degreeCode << course_number << course_name;
			std::string complete_course_name{ output_stream.str() };
			output_stream.str(""); //clear stream content
			course_inputs.push_back(complete_course_name);

		}
	} while (not_finished);

	// Print out full list of courses
	

	if (!course_inputs.empty()) {
		std::cout << "List of courses entered:";

		//for (auto iterator = course_inputs.begin(); iterator != course_inputs.end(); iterator++) {
			//std::cout << *iterator << std::endl;
		//}

		//not much point in splitting, can sort from individual characters easier just to say first int is always same length keep it simplee
		//just use sort (vect.begin, vect.end, sort function) from 
		//std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

		//std::cout << "Select a year. Input a digit to search for that year, e.g. 4 for 4th year courses. X to skip.\n"; //need to select for year, can use if statement with 5th character input[4], ie at input[ degreecode.length]
		std::cin >> input;

		if (std::cin.good()) {
			//just need the first digit
			std::cout << input;
			input = input;
			std::cout << input;

			for (int i{ 0 }; i << course_inputs.size(); i++) {
				std::cout << i;
				if (course_inputs[i][degreeCode.size()] = input) {
					course_specific_year.push_back(course_inputs[i]); // Extract courses belonging to a certain year
				}
			}
			
		}
		else {
			std::cin.clear();
			std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

		}
		 //sort by number or name
		std::sort(course_specific_year.begin(), course_specific_year.end()); //sorts by entire name, no need to direct to just the integer as this comes first.

		for (auto iterator = course_specific_year.begin(); iterator != course_specific_year.end(); iterator++) {
			std::cout << *iterator << std::endl;
		}
		/*start sortinbg, inputs are stored as strings so characters are indexed.


		
		std::string year_choice;
		std::cout << "Please enter year: ";
		std::cin >> year_choice;


		
		*/
	}


	return 0;
}


/*

using namespace std;

int main()
{// sorting numbers and stringsss
	// Warning this type of initialization requires a C++11 Compiler
	vector<int> intVec = { 56, 32, -43, 23, 12, 93, 132, -154 };
	vector<string> stringVec = { "John", "Bob", "Joe", "Z", "Randy" };

	// Sorting the int vector
	sort(intVec.begin(), intVec.end());

	for (vector<int>::size_type i = 0; i != intVec.size(); ++i)
		cout << intVec[i] << " ";

	cout << endl;

	// Sorting the string vector
	sort(stringVec.begin(), stringVec.end());

	// Ranged Based loops. This requires a C++11 Compiler also
	// If you don't have a C++11 Compiler you can use a standard
	// for loop to print your vector.
	for (string& s : stringVec)
		cout << s << " ";

	return 0;
}
*/
