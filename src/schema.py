from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class Ingredient(BaseModel):
    name: str = Field(..., min_length=1, description="Tên nguyên liệu")
    quantity: str = Field(..., min_length=1, description="Định lượng dạng text")
    unit: Optional[str] = Field(None, description="Đơn vị nếu tách được")


class Dish(BaseModel):
    dish_name: Optional[str] = Field(None)
    cuisine: Optional[str] = None
    ingredients: List[Ingredient] = Field(default_factory=list, min_items=0)
    notes: Optional[List[str]] = None


DISH_JSON_SCHEMA = Dish.model_json_schema()