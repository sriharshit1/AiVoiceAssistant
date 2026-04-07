"""
main.py — Launch AURA

  Web UI mode (default):  python main.py
  CLI voice mode:         python main.py --cli
"""

import sys
import os 

if "--cli" in sys.argv:
    from assistant import VoiceAssistant
    VoiceAssistant().run()
else:
    import uvicorn
    import config
    print(f"\n🚀  AURA Web UI → http://localhost:{config.SERVER_PORT}\n")
    uvicorn.run(
        "server:app",
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        reload=False,
        log_level="info",
    )