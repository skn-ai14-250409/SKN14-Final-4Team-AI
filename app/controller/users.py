# controller/users.py

from typing import Union
from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordBearer

router = APIRouter(
    prefix='/users',
    tags=["users"],
    responses={404: {"description": "Not found"}}
)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

def get_current_user(token: str = Depends(oauth2_scheme)):
    return {"user_id": "the current user"}

@router.get("/me")
def read_user_me(current_user = Depends(get_current_user)):
    return current_user

@router.get("/{user_id}")
def read_user(user_id: int):
    return {"user_id": user_id}

