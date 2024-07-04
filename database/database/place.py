import sqlalchemy

from .db_session import SqlAlchemyBase


class Place(SqlAlchemyBase):
    __tablename__ = 'places'

    id = sqlalchemy.Column(
        sqlalchemy.Integer,
        primary_key=True,
        unique=True,
        nullable=False
    )
    number_of_animals = sqlalchemy.Column(
        sqlalchemy.Integer,
        nullable=False,
        default=0
    )
    frame_base64 = sqlalchemy.Column(
        sqlalchemy.String,
        nullable=False
    )
