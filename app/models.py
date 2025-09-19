from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.sql.functions import func

from .database import Base


class SearchHistory(Base):
    __tablename__ = "search_history"

    id           = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_Id      = Column(Integer)
    look_style   = Column(String)
    look_img_url = Column(String, nullable=True)
    look_desc    = Column(Text, nullable=True)
    searched_at  = Column(DateTime(timezone=True), server_default=func.now())

class SearchHistoryProduct(Base):
    __tablename__ = "search_history_product"

    id           = Column(Integer, primary_key=True, index=True, autoincrement=True)
    product_id   = Column(Integer)
    search_id    = Column(Integer)



class ChatHistory(Base):
    __tablename__ = "apiapp_chathistory"

    id            = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id       = Column(Integer)
    influencer_id = Column(Integer)
    talker_type   = Column(String)
    style_text    = Column(Text)
    optional_text = Column(Text, nullable=True)
    voice_url     = Column(String, nullable=True)
    talked_at     = Column(DateTime(timezone=True), server_default=func.now())

