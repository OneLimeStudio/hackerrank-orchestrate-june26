import sys
sys.path.append('code')
from typing import Literal
from pydantic import BaseModel
from google.genai import types

class TestOutput(BaseModel):
    issue_type: Literal['dent', 'scratch']
    evidence_standard_met: bool

try:
    # See if the SDK can handle this Pydantic class
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=TestOutput,
    )
    print('Pydantic Literal handled correctly')
except Exception as e:
    print('Failed with Literal:', e)

