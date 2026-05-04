from dotenv import load_dotenv
load_dotenv()

import os
from openai import OpenAI
c = OpenAI(base_url=os.getenv("QCHEM_BASE_URL"), api_key=os.getenv("QCHEM_API_KEY"))
for m in c.models.list().data: print(m.id)
