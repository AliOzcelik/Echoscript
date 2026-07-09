from sqlalchemy import create_engine, String, Float, Text, event, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from datetime import datetime, timezone
from pathlib import Path


project_root = Path(__file__).resolve().parent.parent
db_path = Path(f"{project_root}/data/notes.db")
db_path.parent.mkdir(parents=True, exist_ok=True)


# the pool may hand a conn to the worker thread (comment added for connect_args)
engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})


# every new connection gets these new pragmas
@event.listen_for(engine, "connect")
def sqlite_pragmas(db_api_conn, _):
    cur = db_api_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL") # 1 writer + many readers, no blocking
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


session = sessionmaker(bind=engine, expire_on_commit=False)

def now():
    return datetime.now(timezone.utc).isoformat()


class Base(DeclarativeBase):
    pass

class Job(Base):
    __tablename__ = "jobs"
    id = mapped_column(String, primary_key=True)
    status = mapped_column(String, defalut="queued", index=True)
    wav_path = mapped_column(String)
    transcript = mapped_column(Text)
    summary = mapped_column(Text)
    error = mapped_column(Text)
    language = mapped_column(String)
    duration_s = mapped_column(Float)
    created_at = mapped_column(String, default=now)
    updated_at = mapped_column(String, default=now, onupdate=now)

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


updatable = {"status", "transcript", "summary", "error", "language", "duration_s"}

def init_db():
    Base.metadata.create_all(engine)

def create_job(job_id, wav_path):
    with Session as s:
        s.add(Job(id = job_id, wav_path = str(wav_path)))
        s.commit()

def update_job(job_id, **fields):
    bad = set(fields) - updatable
    if bad:
        raise ValueError(f"Can't update columns: {bad}")
    with Session() as s:
        job = s.get(Job, job_id)
        if job is None:
            return
        for k, v in fields.items():
            setattr(job, k, v)
        s.commit()

def get_job(job_id):
    with Session() as s:
        job = s.get(Job, job_id)
    if job:
        return job.as_dict()
    return None


def list_jobs():
    with Session() as s:
        rows = s.execute(select(Job).order_by(Job.created_at.desc())).scalars().all()
        return [j.as_dict() for j in rows]
