"""SQLAlchemy models and session setup.

DATABASE_URL controls the backend:
  - unset / sqlite:///ev_charging.db  -> local SQLite (zero setup, used for dev and tests)
  - mysql+pymysql://user:pass@host/db -> MySQL in production

The schema and queries are identical either way; only the connection string changes.
"""
import os
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

Base = declarative_base()


class Station(Base):
    __tablename__ = "stations"
    id = Column(String(16), primary_key=True)
    name = Column(String(128), nullable=False)
    location = Column(String(128), nullable=False)
    capacity = Column(Float, nullable=False)

    readings = relationship("Reading", back_populates="station", cascade="all, delete-orphan")


class Reading(Base):
    __tablename__ = "readings"
    id = Column(Integer, primary_key=True, autoincrement=True)
    station_id = Column(String(16), ForeignKey("stations.id"), nullable=False)
    hour = Column(Integer, nullable=False)
    load_kw = Column(Float, nullable=False)
    utilization = Column(Float, nullable=False)
    risk = Column(String(16), nullable=False)
    recorded_at = Column(DateTime, default=datetime.utcnow)

    station = relationship("Station", back_populates="readings")


class ForecastRun(Base):
    __tablename__ = "forecast_runs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    station_id = Column(String(16), ForeignKey("stations.id"), nullable=False)
    predicted_kw = Column(Float, nullable=False)
    mae = Column(Float)
    rmse = Column(Float)
    mape = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


def get_engine(database_url=None):
    default_sqlite = "/tmp/ev_charging.db" if os.environ.get("VERCEL") else "ev_charging.db"
    url = database_url or os.environ.get("DATABASE_URL", f"sqlite:///{default_sqlite}")
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args)


def init_db(engine):
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def upsert_stations(session, stations):
    for s in stations:
        existing = session.get(Station, s["id"])
        if existing is None:
            session.add(Station(id=s["id"], name=s["name"], location=s["location"], capacity=s["capacity"]))
        else:
            existing.name, existing.location, existing.capacity = s["name"], s["location"], s["capacity"]
    session.commit()


def record_readings(session, hour, poll_results):
    for r in poll_results:
        session.add(Reading(
            station_id=r["id"], hour=hour, load_kw=r["current_load"],
            utilization=r["utilization"], risk=r["risk"],
        ))
    session.commit()


def record_forecast(session, station_id, predicted_kw, metrics=None):
    metrics = metrics or {}
    session.add(ForecastRun(
        station_id=station_id, predicted_kw=predicted_kw,
        mae=metrics.get("mae"), rmse=metrics.get("rmse"), mape=metrics.get("mape"),
    ))
    session.commit()
