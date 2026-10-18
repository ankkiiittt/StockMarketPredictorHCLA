"""
Hugging Face Spaces Entry Point
This file forwards execution to frontend.py for compatibility across default configurations.
"""
import runpy

if __name__ == "__main__":
    runpy.run_path("frontend.py", run_name="__main__")
else:
    import frontend
