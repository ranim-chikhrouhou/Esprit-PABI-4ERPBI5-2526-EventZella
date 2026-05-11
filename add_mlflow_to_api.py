"""
Script to automatically add MLflow endpoints to FastAPI main.py
Run this to integrate MLflow with your API
"""
from pathlib import Path

API_FILE = Path("ML/api/main.py")

# Code to add
IMPORT_LINE = "from ML.api.mlflow_endpoints import router as mlflow_router\n"
ROUTER_LINE = "app.include_router(mlflow_router)\n"

def add_mlflow_to_api():
    """Add MLflow endpoints to FastAPI"""
    
    if not API_FILE.exists():
        print(f"❌ Error: {API_FILE} not found!")
        return False
    
    # Read current content
    content = API_FILE.read_text(encoding="utf-8")
    
    # Check if already added
    if "mlflow_router" in content:
        print("✅ MLflow endpoints already added to FastAPI!")
        return True
    
    lines = content.split("\n")
    new_lines = []
    import_added = False
    router_added = False
    
    for i, line in enumerate(lines):
        new_lines.append(line)
        
        # Add import after other imports
        if not import_added and line.startswith("from ML.api.") and "import" in line:
            # Add after the last ML.api import
            if i + 1 < len(lines) and not lines[i + 1].startswith("from ML.api."):
                new_lines.append(IMPORT_LINE.rstrip())
                import_added = True
        
        # Add router after app creation
        if not router_added and "app = FastAPI(" in line:
            # Find the end of FastAPI initialization
            j = i
            while j < len(lines) and ")" not in lines[j]:
                j += 1
            # Add router after app creation block
            if j < len(lines):
                # Skip to next non-empty line
                while j + 1 < len(lines) and lines[j + 1].strip() == "":
                    new_lines.append(lines[j + 1])
                    j += 1
                # Add router
                new_lines.append("")
                new_lines.append("# MLflow integration")
                new_lines.append(ROUTER_LINE.rstrip())
                router_added = True
                # Skip the lines we already added
                for k in range(i + 1, j + 1):
                    if k < len(lines):
                        lines[k] = None  # Mark as processed
    
    # Filter out None values
    new_lines = [line for line in new_lines if line is not None]
    
    if not import_added or not router_added:
        print("⚠️  Warning: Could not find appropriate location to add MLflow code")
        print("   Please add manually:")
        print()
        print("   1. Add import:")
        print(f"      {IMPORT_LINE.strip()}")
        print()
        print("   2. Add router after app creation:")
        print(f"      {ROUTER_LINE.strip()}")
        return False
    
    # Write back
    API_FILE.write_text("\n".join(new_lines), encoding="utf-8")
    
    print("✅ MLflow endpoints added to FastAPI successfully!")
    print()
    print("📝 Changes made:")
    print(f"   - Added import: {IMPORT_LINE.strip()}")
    print(f"   - Added router: {ROUTER_LINE.strip()}")
    print()
    print("🔄 Next steps:")
    print("   1. Restart FastAPI server")
    print("   2. Check http://localhost:8000/docs")
    print("   3. Look for 'MLflow' section in API docs")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("🔧 Adding MLflow Endpoints to FastAPI")
    print("=" * 60)
    print()
    
    success = add_mlflow_to_api()
    
    if success:
        print()
        print("=" * 60)
        print("✅ Integration Complete!")
        print("=" * 60)
    else:
        print()
        print("=" * 60)
        print("❌ Integration Failed - Manual Setup Required")
        print("=" * 60)
        print()
        print("See N8N_MLFLOW_INTEGRATION.md for manual setup instructions")
