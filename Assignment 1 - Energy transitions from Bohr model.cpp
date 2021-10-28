// Assignment 1 - Energy transitions from Bohr model
// Jonathan Ryding 31/01/2020
// Program to calculate a transition energy using Bohr model, validating user input.

#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <sstream> //stringstream
#include <ctype.h> //for checking digits user input validation
#include <limits>

double calculate_transition_energy(int atomic_number, int initial_quantum_number, int final_quantum_number, bool convert_to_joules)
{ //function to calculate transition energy from input parameter and output in desired eV or J
    double transition_energy;
    transition_energy = 13.6 * pow(atomic_number, 2) * (1 / pow(final_quantum_number, 2) - 1 / pow(initial_quantum_number, 2));

    if (convert_to_joules == true)
        transition_energy = transition_energy * 1.6 * pow(10, -19); //conversion

    return transition_energy;
}

int string_input_to_integer(std::string input) 
{ //function to return a valid integer 0,1,2,3... from a string input from user
    bool is_integer_valid{ false };
    int valid_char_counter{ };
    int valid_integer;
 //use a counter, if the counted number of valid digits is the same as the length of the input string, accept input.

    while (is_integer_valid == false) {

        std::cout << "Enter input. \n";
        std::cin >> input; 

        if (input[0] == '0' && isdigit(input[1]) || std::cin.peek() != '\n' ) { //peek previews next character without removing it, for valid input the last character is a return. If statement skips counter.
        } 
        else {
            for (int i = 0; i < input.size(); i++) { 
                if (isdigit(input[i]))
                    valid_char_counter++;
            }
        }
        if (valid_char_counter == input.size()) {
            valid_integer = std::stoi(input); 
            is_integer_valid = true; //returns a number 0,1,2,3...
        }
        else {
            std::cout << "Invaild input\n";
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
            //numeric_limits<streamsize>::max() is used to clear the stream up until the last return.
            valid_char_counter = 0;
        }
    }
    return valid_integer;
}

int main()
{
    std::string atomic_number = "";
    int valid_atomic_number;
    std::string initial_quantum_number = "";
    int valid_initial_quantum_number;
    std::string final_quantum_number = "";
    int valid_final_quantum_number;

    double transition_energy;
    bool convert_to_joules;

    char input_repeat;
    char input_convert;
    bool repeat{ true };

    std::cout << "Program to calculate transition energy using Bohr model\n";

    while (repeat) {
        // Ask user to enter atomic number, initial and final quantum numbers.
        std::cout << "Input the required atomic number:\n";
        valid_atomic_number = string_input_to_integer(atomic_number);

        while (valid_atomic_number > 118) {
            std::cout << "Atomic number was greater than 118.\n";
            valid_atomic_number = string_input_to_integer(atomic_number);
        }
        while (valid_atomic_number == 0) {
            std::cout << "Atomic number cannot be 0.\n";
            valid_atomic_number = string_input_to_integer(atomic_number);
        }
        std::cout << "Input accepted\n";

        std::cout << "Input the required initial quantum number:\n";
        valid_initial_quantum_number = string_input_to_integer(initial_quantum_number);
        while (valid_initial_quantum_number == 0) {
            std::cout << "Initial quantum number cannot be 0.\n";
            valid_initial_quantum_number = string_input_to_integer(initial_quantum_number);
        }
        std::cout << "Input accepted\n";

        std::cout << "Input the required final quantum number:\n";
        valid_final_quantum_number = string_input_to_integer(final_quantum_number);
        while (valid_final_quantum_number == 0) {
            std::cout << "Final quantum number cannot be 0.\n";
            valid_final_quantum_number = string_input_to_integer(final_quantum_number);
        }
        std::cout << "Input accepted\n";

        std::cout << "Change output from eV to J? (y/n)\n";
        std::cin >> input_convert; 

        while (!((input_convert == 'y') || input_convert == 'n')) { //while first character isn't y or n, repeat request whether to convert

            std::cout << "Enter y/n\n";
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cin >> input_convert;
        }

        if (input_convert == 'y') {
            convert_to_joules = true;
            transition_energy = calculate_transition_energy(valid_atomic_number, valid_initial_quantum_number, valid_final_quantum_number, convert_to_joules); // Compute photon energy, Delta E = 13.6*(Z^2)*(1/n_j^2-1/n_i^2) eV
            std::cout << "The transition energy is " << std::setprecision(3) << transition_energy << " J" << "\nRepeat? (y/n)"; //set precision to three places
            
        }
        if (input_convert == 'n') {
            convert_to_joules = false;
            transition_energy = calculate_transition_energy(valid_atomic_number, valid_initial_quantum_number, valid_final_quantum_number, convert_to_joules);
            std::cout << "The transition energy is " << std::setprecision(3) << transition_energy << " eV" << "\nRepeat? (y/n)";
            
        }
        std::cin >> input_repeat;
        while (!((input_repeat == 'y') || input_repeat == 'n')) { //while first character isn't y or n, repeat request whether to repeat program
          
            std::cout << "Enter y/n\n";
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cin >> input_repeat;
        }
        if (input_repeat == 'n') {
            repeat = false;
        }
    }
    return 0;
}