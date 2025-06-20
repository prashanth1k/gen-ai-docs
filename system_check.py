#!/usr/bin/env python3
"""
System Resource Checker for gen-ai-docs
Run this before processing large PDFs to ensure system stability.
"""

import psutil
import sys
import os

def check_system_resources():
    """Check if system has sufficient resources to run gen-ai-docs safely."""
    
    print("🔍 Checking system resources for gen-ai-docs...")
    print()
    
    # Check available memory
    memory = psutil.virtual_memory()
    print(f"💾 System Memory:")
    print(f"   Total: {memory.total / (1024**3):.1f} GB")
    print(f"   Used: {memory.used / (1024**3):.1f} GB ({memory.percent:.1f}%)")
    print(f"   Available: {memory.available / (1024**3):.1f} GB")
    
    # Check available disk space
    disk = psutil.disk_usage('.')
    print(f"💿 Disk Space (current directory):")
    print(f"   Total: {disk.total / (1024**3):.1f} GB")
    print(f"   Used: {disk.used / (1024**3):.1f} GB ({(disk.used/disk.total)*100:.1f}%)")
    print(f"   Free: {disk.free / (1024**3):.1f} GB")
    
    # Check CPU
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"⚡ CPU:")
    print(f"   Cores: {cpu_count}")
    print(f"   Current Usage: {cpu_percent:.1f}%")
    
    print()
    
    # Safety checks
    warnings = []
    errors = []
    
    if memory.available / (1024**3) < 2.0:
        errors.append(f"❌ CRITICAL: Only {memory.available / (1024**3):.1f}GB RAM available (need at least 2GB)")
    elif memory.available / (1024**3) < 4.0:
        warnings.append(f"⚠️  WARNING: Only {memory.available / (1024**3):.1f}GB RAM available (recommended: 4GB+)")
    
    if memory.percent > 85:
        errors.append(f"❌ CRITICAL: System memory usage at {memory.percent:.1f}% (too high)")
    elif memory.percent > 70:
        warnings.append(f"⚠️  WARNING: System memory usage at {memory.percent:.1f}% (may cause slowdowns)")
    
    if disk.free / (1024**3) < 1.0:
        warnings.append(f"⚠️  WARNING: Only {disk.free / (1024**3):.1f}GB disk space free")
    
    if cpu_percent > 90:
        warnings.append(f"⚠️  WARNING: High CPU usage ({cpu_percent:.1f}%)")
    
    # Show results
    if errors:
        print("🚨 SYSTEM NOT SAFE FOR PROCESSING:")
        for error in errors:
            print(f"   {error}")
        print()
        print("❌ Please close other applications and free up memory before running gen-ai-docs.")
        return False
    
    if warnings:
        print("⚠️  SYSTEM WARNINGS:")
        for warning in warnings:
            print(f"   {warning}")
        print()
        print("💡 Consider closing other applications for best performance.")
    
    print("✅ System appears safe for gen-ai-docs processing.")
    print()
    
    # Recommendations
    print("📋 RECOMMENDATIONS:")
    print("   • Use default mode (fastest): python main.py process --input-pdf your_file.pdf")
    print("   • For large files (>50MB): monitor system during processing")
    print("   • If crashes occur: reduce PDF size or use cloud processing")
    print()
    
    return True

def check_file_size(pdf_path):
    """Check PDF file size and provide recommendations."""
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return False
    
    size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    print(f"📄 PDF File: {pdf_path}")
    print(f"📏 Size: {size_mb:.1f} MB")
    
    if size_mb > 100:
        print(f"⚠️  WARNING: Large file ({size_mb:.1f}MB) may require significant memory")
        print("   Consider using default mode for fastest processing")
    elif size_mb > 50:
        print(f"💡 Moderate size file ({size_mb:.1f}MB) - should process fine")
    else:
        print(f"✅ Small file ({size_mb:.1f}MB) - should process quickly")
    
    print()
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        print("🚀 gen-ai-docs System Safety Check")
        print("=" * 50)
        print()
        
        if check_file_size(pdf_path):
            if check_system_resources():
                print("🎯 Ready to process! Run:")
                print(f"   python main.py process --input-pdf \"{pdf_path}\"")
            else:
                sys.exit(1)
        else:
            sys.exit(1)
    else:
        print("🚀 gen-ai-docs System Safety Check")
        print("=" * 50)
        print()
        
        if check_system_resources():
            print("🎯 System ready for processing!")
        else:
            sys.exit(1) 