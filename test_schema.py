import sys
sys.path.append('code')
from pydantic import BaseModel
from google.genai import types
from schemas import CarClaimOutput

schema = CarClaimOutput.model_json_schema()
print('Generated schema keys:', schema.keys())
try:
    types.Schema(**schema)
    print('GenAI Schema loaded successfully')
except Exception as e:
    print('Failed:', e)
