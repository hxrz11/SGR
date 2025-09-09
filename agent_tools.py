import os
import re
from typing import Any, Dict, List, Tuple

import httpx

from database import DatabaseManager

# Global database manager instance
_db_manager = DatabaseManager()

async def run_sql(sql: str) -> Tuple[List[Dict[str, Any]], str]:
    """Execute a SELECT query against PurchaseAllView with safety checks.

    - Only allows SELECT statements targeting "PurchaseAllView".
    - Ensures the filter "PurchaseRecordStatus"='A'.
    - Ensures a LIMIT 100 is applied when none provided.

    Returns the query results and the final executed SQL.
    """
    if not re.match(r"^\s*select", sql, re.IGNORECASE):
        raise ValueError("Only SELECT queries are permitted")

    if "purchaseallview" not in sql.lower():
        raise ValueError('Queries must reference "PurchaseAllView"')

    # Remove trailing semicolon for consistent manipulation
    query = sql.strip().rstrip(";")

    # Ensure the PurchaseRecordStatus filter is applied
    if re.search(r"where", query, re.IGNORECASE):
        if not re.search(r"purchaserecordstatus", query, re.IGNORECASE):
            query = re.sub(
                r"where",
                'WHERE "PurchaseRecordStatus"=\'A\' AND',
                query,
                flags=re.IGNORECASE,
                count=1,
            )
    else:
        query += ' WHERE "PurchaseRecordStatus"=\'A\''

    # Ensure LIMIT 100 is present
    if not re.search(r"limit\s+\d+", query, re.IGNORECASE):
        query += " LIMIT 100"

    # Initialize connection pool on first use
    if _db_manager.pool is None:
        await _db_manager.initialize()

    return await _db_manager.execute_query(query)

async def call_status_api(purchase_card_id: str) -> Dict[str, Any]:
    """Fetch status timeline for a purchase from an external API."""
    base_url = os.getenv("STATUS_API_URL", "http://localhost:8000")
    url = f"{base_url}/status/{purchase_card_id}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()

def clarification(questions: List[str]) -> Dict[str, Any]:
    """Prepare a structure for clarification questions."""
    return {"clarification_needed": True, "questions": questions}

def report_completion(answer: str, steps: List[str]) -> Dict[str, Any]:
    """Form the final answer along with a short list of steps taken."""
    return {"answer": answer, "steps": steps}
