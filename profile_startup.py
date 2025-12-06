import cProfile
import pstats
import sys
import os
import shutil

# Ensure we can import from src
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, 'src'))

# Mock line_profiler's profile decorator before importing main_gui
import builtins
if not hasattr(builtins, 'profile'):
    def profile(func):
        return func
    builtins.profile = profile

try:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QTimer
    from src.main_gui import PortfolioApp
    import src.config as config
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.path.append(os.path.abspath("src"))
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QTimer
    from src.main_gui import PortfolioApp
    import src.config as config

def run_profiling():
    print("Starting profiling...")
    
    # Create QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    app.setOrganizationName("StockAlchemist")
    app.setApplicationName(config.APP_NAME)
    
    # Start Profiler
    profiler = cProfile.Profile()
    profiler.enable()
    
    print("Initializing PortfolioApp...")
    window = PortfolioApp()
    
    # Optional: trigger load if it doesn't happen in init
    # window.refresh_data() # Example if needed
    
    print("App initialized. Waiting 20 seconds to capture startup activity...")
    
    def stop_profiling():
        print("Stopping profiling...")
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.strip_dirs().sort_stats('cumtime').dump_stats('startup_profile.prof')
        
        # Also print top 20 cumulative time
        stats.print_stats(20)
        
        print("Profiling finished. Results saved to startup_profile.prof")
        app.quit()
        
    # Stop after 20 seconds
    QTimer.singleShot(20000, stop_profiling)
    
    # Run the event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    run_profiling()
