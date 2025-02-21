from fastapi import APIRouter
from .routers import user, list

router = APIRouter()

router.include_router(user.user_router)
# router.include_router(identificaciones.router)
router.include_router(list.router)
