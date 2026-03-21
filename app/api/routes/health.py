"""
Health check endpoints.
"""

import logging
from fastapi import APIRouter

from app.config import settings
from app.schemas.api import HealthResponse, PistonHealthResponse
from app.clients.piston_client import get_piston_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Overall service health check.
    
    Returns status of all dependent services.
    """
    services = {}
    
    # Check PISTON
    try:
        piston = await get_piston_client()
        services["piston"] = await piston.health_check()
    except Exception as e:
        logger.warning(f"PISTON health check failed: {e}")
        services["piston"] = False
    
    # Check Supabase (simple connectivity)
    try:
        from app.clients.supabase_client import get_supabase_client
        db = get_supabase_client()
        # Actually query to verify connection
        result = db.client.table("base_questions").select("id").limit(1).execute()
        services["supabase"] = result.data is not None
    except Exception as e:
        logger.warning(f"Supabase health check failed: {e}")
        services["supabase"] = False
    
    # Determine overall status
    all_healthy = all(services.values())
    any_healthy = any(services.values())
    
    if all_healthy:
        status = "healthy"
    elif any_healthy:
        status = "degraded"
    else:
        status = "unhealthy"
    
    return HealthResponse(
        status=status,
        version=settings.APP_VERSION,
        services=services
    )


@router.get("/piston", response_model=PistonHealthResponse)
async def piston_health() -> PistonHealthResponse:
    """
    PISTON API specific health check.
    
    Returns available languages and runtime versions.
    """
    try:
        piston = await get_piston_client()
        is_healthy = await piston.health_check()
        
        if is_healthy:
            return PistonHealthResponse(
                available=True,
                languages=["python", "java", "cpp"]
            )
        else:
            return PistonHealthResponse(
                available=False,
                error="PISTON API not responding"
            )
    except Exception as e:
        return PistonHealthResponse(
            available=False,
            error=str(e)
        )


@router.get("/ready")
async def readiness_check():
    """
    Kubernetes-style readiness probe.
    
    Returns 200 if service can handle requests.
    """
    health = await health_check()
    
    if health.status == "unhealthy":
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {"ready": True}


@router.get("/live")
async def liveness_check():
    """
    Kubernetes-style liveness probe.
    
    Returns 200 if service is running.
    """
    return {"alive": True}


@router.post("/prewarm")
async def prewarm_runtimes():
    """
    Manually prewarm Java JVM and C++ compiler.
    
    Call this if you're getting timeout errors after Docker restart.
    """
    results = {}
    
    prewarm_codes = {
        "python": "print('warm')",
        "java": """
public class Main {
    public static void main(String[] args) {
        System.out.println("warm");
    }
}
""",
        "cpp": """
#include <iostream>
int main() {
    std::cout << "warm" << std::endl;
    return 0;
}
"""
    }
    
    try:
        piston = await get_piston_client()
        
        for lang, code in prewarm_codes.items():
            try:
                result = await piston.execute(lang, code, "")
                results[lang] = {
                    "success": result.success,
                    "output": result.stdout if result.success else result.error_message
                }
            except Exception as e:
                results[lang] = {"success": False, "error": str(e)}
        
        return {"prewarmed": True, "results": results}
    
    except Exception as e:
        return {"prewarmed": False, "error": str(e)}
