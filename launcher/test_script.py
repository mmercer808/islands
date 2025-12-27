"""Test script to demonstrate PyLauncher output capture."""
import sys
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    print("Hello from PyLauncher test script!")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {__file__}")
    print()
    
    # Test stdout
    for i in range(5):
        print(f"Processing item {i + 1}/5...")
        time.sleep(0.3)
    
    print()
    
    # Test logging
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.debug("This is a debug message")
    
    print()
    
    # Test stderr
    print("Writing to stderr...", file=sys.stderr)
    
    print()
    print("✅ Test completed successfully!")

if __name__ == "__main__":
    main()
