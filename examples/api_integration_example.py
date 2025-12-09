#!/usr/bin/env python3
"""
API Integration Example

This script demonstrates how to integrate with the Supply Chain Smart Replenishment System API to:
1. Get real-time demand forecasts
2. Get real-time replenishment plans based on forecasts and inventory data

Usage:
    python api_integration_example.py --product_id 1 --forecast_days 7
"""

import requests
import json
import argparse
from datetime import datetime, timedelta

# API base URL
BASE_URL = "http://localhost:8000"

def get_real_time_forecast(product_id, forecast_days=7):
    """
    Get real-time demand forecast from API
    
    Args:
        product_id: Product ID
        forecast_days: Number of days to forecast
        
    Returns:
        forecast_result: Dictionary containing forecast data
    """
    endpoint = f"{BASE_URL}/api/forecast/real-time"
    
    params = {
        "product_id": product_id,
        "forecast_days": forecast_days
    }
    
    try:
        response = requests.post(endpoint, params=params)
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting forecast: {e}")
        print(f"Response content: {response.text if 'response' in locals() else 'N/A'}")
        return None

def get_real_time_optimization(product_id, forecast_results, inventory_data, lead_times, costs, constraints=None):
    """
    Get real-time replenishment plan optimization from API
    
    Args:
        product_id: Product ID
        forecast_results: Forecast results from get_real_time_forecast()
        inventory_data: Current inventory data
        lead_times: Lead time data
        costs: Cost parameters
        constraints: Constraint parameters (optional)
        
    Returns:
        optimization_result: Dictionary containing optimization plan
    """
    endpoint = f"{BASE_URL}/api/optimize/real-time"
    
    # Convert forecast results to the format expected by the optimization endpoint
    formatted_forecast_results = [
        {
            "product_id": product_id,
            "predictions": forecast_results["forecast"]
        }
    ]
    
    payload = {
        "product_id": product_id,
        "forecast_results": formatted_forecast_results,
        "inventory_data": inventory_data,
        "lead_times": lead_times,
        "costs": costs
    }
    
    # Add constraints if provided
    if constraints:
        payload["constraints"] = constraints
    
    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting optimization plan: {e}")
        return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Supply Chain API Integration Example")
    parser.add_argument("--product_id", type=int, default=1, help="Product ID to forecast")
    parser.add_argument("--forecast_days", type=int, default=7, help="Number of days to forecast")
    args = parser.parse_args()
    
    print("=" * 50)
    print("Supply Chain API Integration Example")
    print("=" * 50)
    
    # Step 1: Get real-time forecast
    print(f"\n1. Getting real-time forecast for product {args.product_id} ({args.forecast_days} days)...")
    forecast_result = get_real_time_forecast(args.product_id, args.forecast_days)
    
    if forecast_result:
        print("Forecast result:")
        print(f"   Product ID: {forecast_result['product_id']}")
        print(f"   Forecast days: {forecast_result['forecast_days']}")
        print(f"   Model used: {forecast_result['model_used']}")
        print(f"   Forecast start date: {forecast_result['forecast_start_date']}")
        print("   Forecast values:", forecast_result['forecast'])
    else:
        print("Failed to get forecast")
        return
    
    # Step 2: Prepare data for optimization
    print("\n2. Preparing data for optimization...")
    
    # Example inventory data
    inventory_data = {
        args.product_id: {
            "current_inventory": 100,
            "safety_stock": 20,
            "reorder_point": 50
        }
    }
    
    # Example lead times (days)
    lead_times = {
        args.product_id: 3
    }
    
    # Example cost parameters
    costs = {
        "ordering_cost": [100],  # Per order cost
        "holding_cost": [0.5],    # Per unit per period holding cost
        "unit_cost": [10]         # Per unit cost
    }
    
    # Example constraints (optional)
    constraints = {
        "budget_constraint": 5000,  # Maximum budget per order
        "max_order_quantity": 200,  # Maximum order quantity per product
        "min_order_quantity": 10    # Minimum order quantity per product
    }
    
    print("Inventory data:", inventory_data)
    print("Lead times:", lead_times)
    print("Costs:", costs)
    print("Constraints:", constraints)
    
    # Step 3: Get real-time optimization plan
    print(f"\n3. Getting real-time optimization plan for product {args.product_id}...")
    optimization_result = get_real_time_optimization(
        args.product_id, 
        forecast_result, 
        inventory_data, 
        lead_times, 
        costs,
        constraints  # Pass optional constraints
    )
    
    if optimization_result:
        print("Optimization result:")
        # Print the result in a readable format
        print(json.dumps(optimization_result, indent=2, ensure_ascii=False))
    else:
        print("Failed to get optimization plan")
        return
    
    print("\n" + "=" * 50)
    print("Integration complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
