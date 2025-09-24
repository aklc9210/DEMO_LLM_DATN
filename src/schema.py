from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class Ingredient(BaseModel):
    name: str = Field(..., min_length=1, description="Tên nguyên liệu")
    quantity: str = Field(..., min_length=1, description="Định lượng dạng text")
    unit: Optional[str] = Field(None, description="Đơn vị nếu tách được")


class Dish(BaseModel):
    dish_name: str = Field(..., min_length=1)
    cuisine: Optional[str] = None
    ingredients: List[Ingredient] = Field(..., min_items=1)
    notes: Optional[List[str]] = None


@field_validator("ingredients")
@classmethod
def _non_empty(cls, v):
    if not v:
        raise ValueError("ingredients cannot be empty")
    return v


DISH_JSON_SCHEMA = Dish.model_json_schema()