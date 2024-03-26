FROM kitware/trame

COPY --chown=trame-user:trame-user . /deploy

ENV TRAME_CLIENT_TYPE=vue3
RUN /opt/trame/entrypoint.sh build
