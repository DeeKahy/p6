/**
 * Handles age invariants in places
 * 
 * @param condition takes a float and a bool for returning the result
 * @return Invariant uses pass by reference and returns into the bool* parameter
 */
struct Invariant {
    void(*condition)(float*, bool*);
};