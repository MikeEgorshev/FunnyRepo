import sys
from pathlib import Path

# тесты запускаются и без установки пакета: PYTHONPATH не нужен
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
