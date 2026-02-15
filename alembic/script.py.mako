"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | default('None')}
Create Date: ${create_date}
"""

from alembic import op
import sqlalchemy as sa

revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = None
depends_on = None

def upgrade():
    pass

def downgrade():
    pass
