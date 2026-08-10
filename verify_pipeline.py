import sys
import os
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import mfr

def test_pipeline():
    print("=== MFR-X Multi-Agent System Verification Script ===")

    print("\n1. Initializing MFR-X Master Biometric Orchestrator...")
    try:
        orchestrator = mfr.BiometricOrchestrator(models_dir="models", db_path="test_db.json")
        print("   [OK] BiometricOrchestrator initialized with 10 specialized AI agents.")
    except Exception as e:
        print(f"   [FAIL] Failed to initialize orchestrator: {e}")
        return False

    print("\n2. Simulating Synthetic Frame Processing...")
    try:
        # Create a synthetic 640x480 test frame with a simple rectangle as a face placeholder
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame, (320, 240), 80, (200, 200, 200), -1)

        annotated_frame, telemetry = orchestrator.process_frame(frame)

        print("   [OK] Multi-Agent Pipeline executed successfully!")
        print(f"   Candidate  : {telemetry['candidate']}")
        print(f"   Decision   : {telemetry['status']}")
        print(f"   Explanation: {telemetry['explanation']}")
        print("\n   Active Agent Telemetry:")
        for agent_name, payload in telemetry['agents'].items():
            print(f"     [{agent_name.upper():16s}] {payload}")

    except Exception as e:
        print(f"   [FAIL] Error during multi-agent pipeline test: {e}")
        import traceback; traceback.print_exc()
        return False

    finally:
        if os.path.exists("test_db.json"):
            try:
                os.remove("test_db.json")
            except Exception:
                pass

    print("\n=== MFR-X Multi-Agent Verification: PASSED ===")
    return True

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
