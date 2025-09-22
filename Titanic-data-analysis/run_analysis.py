#!/usr/bin/env python3
"""
Titanic Analysis Runner - Execute Complete Analysis Suite
========================================================

This script runs the complete advanced Titanic analysis pipeline.
Usage: python run_analysis.py [--dashboard] [--stats] [--all]
"""

import sys
import argparse
import subprocess
import os

def run_main_analysis():
    """Run the main advanced analysis"""
    print("🚀 Running Advanced Titanic Analysis...")
    try:
        subprocess.run([sys.executable, "advanced_titanic_analysis.py"], check=True)
        print("✅ Main analysis completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Main analysis failed: {e}")
        return False
    return True

def run_statistical_analysis():
    """Run the statistical analysis"""
    print("📊 Running Statistical Analysis...")
    try:
        subprocess.run([sys.executable, "statistical_analysis.py"], check=True)
        print("✅ Statistical analysis completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Statistical analysis failed: {e}")
        return False
    return True

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    print("🌐 Launching Interactive Dashboard...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "titanic_dashboard.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Dashboard launch failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    return True

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Run Titanic Analysis Suite")
    parser.add_argument("--dashboard", action="store_true", help="Launch dashboard only")
    parser.add_argument("--stats", action="store_true", help="Run statistical analysis only")
    parser.add_argument("--all", action="store_true", help="Run all analyses")
    
    args = parser.parse_args()
    
    print("🚢 TITANIC SURVIVAL ANALYSIS SUITE")
    print("=" * 50)
    
    if args.dashboard:
        launch_dashboard()
    elif args.stats:
        run_statistical_analysis()
    elif args.all:
        run_main_analysis()
        run_statistical_analysis()
        print("\n🌐 Launch dashboard? (y/n): ", end="")
        if input().lower().startswith('y'):
            launch_dashboard()
    else:
        # Default: run main analysis
        run_main_analysis()
        print("\n🌐 Launch dashboard? (y/n): ", end="")
        if input().lower().startswith('y'):
            launch_dashboard()

if __name__ == "__main__":
    main()