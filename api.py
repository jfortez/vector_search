from fastapi import APIRouter
from router import user,identificaciones
router = APIRouter()

router.include_router(user.user_router)
router.include_router(identificaciones.router)


