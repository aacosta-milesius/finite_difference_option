/**
 * Black-Scholes equation solver
 * That crazy equation: ∂V/∂t + (1/2)σ²(∂²V/∂S²) + rS(∂V/∂S) - rV = 0
 * 
 * using finite difference method
 * 
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <iomanip>

class BlackScholesPDE {
private:
    double sigma;  // How jumpy the stock is
    double r;      // Risk-free rate - basically free money
    double T;      // Time till expiry
    double S_max;  // Max stock price we care about
    int M;         // Grid points for stock price
    int N;         // Grid points for time
    double dt;     // Time step size
    double dS;     // Stock price step size
    std::vector<std::vector<double> > grid; // Our solution lives here

public:
    // Constructor - gotta set everything up
    BlackScholesPDE(double sigma, double r, double T, double S_max, int M, int N) 
        : sigma(sigma), r(r), T(T), S_max(S_max), M(M), N(N) {
        dt = T / N;
        dS = S_max / M;
        
        // Make the grid - start with all zeros
        grid.resize(M+1, std::vector<double>(N+1, 0.0));
    }

    // Set the payoff at expiry
    void setPayoff(double K, bool isCall = true) {
        for (int i = 0; i <= M; i++) {
            double S = i * dS;
            if (isCall)
                grid[i][N] = std::max(S - K, 0.0); // Call payoff - max(S-K, 0)
            else
                grid[i][N] = std::max(K - S, 0.0); // Put payoff - max(K-S, 0)
        }
    }

    // Set the edge conditions
    void setBoundaryConditions(double K, bool isCall = true) {
        // When stock price = 0
        for (int j = 0; j <= N; j++) {
            if (isCall)
                grid[0][j] = 0.0; // Call worth zero if stock is worth zero
            else
                grid[0][j] = K * exp(-r * (T - j*dt)); // Put worth discounted K
        }
        
        // When stock price = max
        for (int j = 0; j <= N; j++) {
            if (isCall)
                grid[M][j] = S_max - K * exp(-r * (T - j*dt)); // Call worth S-K basically
            else
                grid[M][j] = 0.0; // Put worth zero when stock is super high
        }
    }

    // Work backwards to solve the PDE
    void solve() {
        // Go backwards from expiry to now
        for (int j = N-1; j >= 0; j--) {
            for (int i = 1; i < M; i++) {
                double S = i * dS;
                
                // Math stuff for finite difference
                double a = 0.5 * dt * (pow(sigma * S, 2) / pow(dS, 2) - r * S / dS);
                double b = 1.0 - dt * (pow(sigma * S, 2) / pow(dS, 2) + r);
                double c = 0.5 * dt * (pow(sigma * S, 2) / pow(dS, 2) + r * S / dS);
                
                // The actual finite difference step
                grid[i][j] = a * grid[i-1][j+1] + b * grid[i][j+1] + c * grid[i+1][j+1];
            }
        }
    }

    // Get option price for any stock price and time
    double getPrice(double S, double t) const {
        int i = static_cast<int>(S / dS);
        int j = static_cast<int>(t / dt);
        
        // Don't blow up with bad inputs
        if (i < 0 || i >= M || j < 0 || j >= N) {
            throw std::out_of_range("Price or time out of range: S=" + 
                                    std::to_string(S) + ", t=" + std::to_string(t));
        }
        
        // Interpolate between grid points
        double alpha = (S - i * dS) / dS;
        double beta = (t - j * dt) / dt;
        
        double v1 = grid[i][j];
        double v2 = grid[i+1][j];
        double v3 = grid[i][j+1];
        double v4 = grid[i+1][j+1];
        
        return (1 - alpha) * (1 - beta) * v1 + alpha * (1 - beta) * v2 +
               (1 - alpha) * beta * v3 + alpha * beta * v4;
    }

    // Get prices for a range of stock values
    std::vector<std::pair<double, double> > getPriceSlice(double t = 0.0) const {
        std::vector<std::pair<double, double> > prices;
        int j = static_cast<int>(t / dt);
        
        for (int i = 0; i <= M; i++) {
            double S = i * dS;
            prices.push_back(std::make_pair(S, grid[i][j]));
        }
        
        return prices;
    }

    // Print part of the grid - debugging mostly
    void printGrid(int start_i = 0, int end_i = -1, int start_j = 0, int end_j = -1) const {
        // Fix missing end values
        if (end_i < 0) end_i = M;
        if (end_j < 0) end_j = N;
        
        // Keep indexes in bounds
        start_i = std::max(0, std::min(start_i, M));
        end_i = std::max(0, std::min(end_i, M));
        start_j = std::max(0, std::min(start_j, N));
        end_j = std::max(0, std::min(end_j, N));
        
        for (int i = start_i; i <= end_i; i++) {
            double S = i * dS;
            std::cout << "S = " << std::fixed << std::setprecision(2) << S << ": ";
            for (int j = start_j; j <= end_j; j++) {
                std::cout << std::fixed << std::setprecision(6) << grid[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }
};

int main() {
    // Inputs - change these to whatever you want
    double sigma = 0.2;     // Stock volatility - 20%
    double r = 0.05;        // Interest rate - 5%
    double T = 1.0;         // 1 year till expiry
    double S_0 = 100.0;     // Current stock price
    double K = 100.0;       // Strike price
    double S_max = 300.0;   // Max stock price in our grid
    int M = 100;            // Grid points for stock
    int N = 1000;           // Grid points for time
    bool isCall = true;     // Doing a call option

    try {
        // Create, setup and solve
        BlackScholesPDE bs(sigma, r, T, S_max, M, N);
        bs.setPayoff(K, isCall);
        bs.setBoundaryConditions(K, isCall);
        bs.solve();

        // Get the option price right now
        double optionPrice = bs.getPrice(S_0, 0.0);
        
        // Show results
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Option type: " << (isCall ? "Call" : "Put") << "\n";
        std::cout << "Stock price: " << S_0 << "\n";
        std::cout << "Strike price: " << K << "\n";
        std::cout << "Time to maturity: " << T << " year\n";
        std::cout << std::setprecision(6);
        std::cout << "Option price: " << optionPrice << "\n";
        
        // Show a bit of the grid around current price
        int center_i = static_cast<int>(S_0 / S_max * M);
        int start_i = std::max(0, center_i - 2);
        int end_i = std::min(M, center_i + 2);
        std::cout << "\nOption price grid (excerpt):\n";
        bs.printGrid(start_i, end_i, 0, 2);
        
        // Show prices for stock prices near current
        std::cout << "\nOption prices at t=0 for different stock prices:\n";
        std::vector<std::pair<double, double> > prices = bs.getPriceSlice();
        
        // Just grab the interesting ones
        std::vector<std::pair<double, double> > filtered_prices;
        for (size_t i = 0; i < prices.size(); ++i) {
            if (prices[i].first >= 80.0 && prices[i].first <= 120.0) {
                filtered_prices.push_back(prices[i]);
            }
        }
        
        size_t count = std::min(filtered_prices.size(), static_cast<size_t>(10));
        for (size_t i = 0; i < count; ++i) {
            double stock_price = filtered_prices[i].first;
            double option_price = filtered_prices[i].second;
            std::cout << "S = " << std::fixed << std::setprecision(2) 
                      << stock_price << ": " << std::setprecision(6) 
                      << option_price << "\n";
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
} 
