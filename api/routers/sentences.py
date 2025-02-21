from fastapi import APIRouter


router = APIRouter(prefix="/sentences", tags=["sentences"])


@router.get("/")
def get_sentences():
    return {"message": "Hello World"}
