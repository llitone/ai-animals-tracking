import sqlalchemy

from .db_session import SqlAlchemyBase


class Animal(SqlAlchemyBase):
    __tablename__ = 'animals'
    id = sqlalchemy.Column(
        sqlalchemy.Integer,
        primary_key=True,
        unique=True,
        nullable=False
    )
    place_id = sqlalchemy.Column(
        sqlalchemy.Integer,
        sqlalchemy.ForeignKey('places.id')
    )
    object_class = sqlalchemy.Column(
        sqlalchemy.String,
        nullable=False
    )
    first_detection = sqlalchemy.Column(
        sqlalchemy.String,
        nullable=False
    )
    last_detection = sqlalchemy.Column(
        sqlalchemy.String,
        nullable=False
    )
