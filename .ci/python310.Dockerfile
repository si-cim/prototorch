FROM python:3.9

RUN adduser --uid 1000 jenkins

USER jenkins

RUN mkdir -p /home/jenkins/.cache/pip
