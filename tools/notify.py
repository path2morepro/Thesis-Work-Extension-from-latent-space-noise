#!/usr/bin/env python3
"""Send a one-shot notification about a pipeline run.

Two independent channels, both optional; whichever is configured gets used.
Configuration lives OUTSIDE the repo, in ~/.config/thesis-notify.env, so an
app password can never be committed:

    # push to a phone via https://ntfy.sh (install the ntfy app, subscribe to
    # the topic). No account needed. The topic name is the only secret, so keep
    # it long and random -- anyone who knows it can read your notifications.
    NOTIFY_NTFY_TOPIC=thesis-sqg-xxxxxxxxxxxx

    # email, e.g. to read in Gmail or Outlook on a phone. For Gmail this needs
    # an App Password (Google account -> Security -> 2-step verification ->
    # App passwords), NOT the normal account password.
    NOTIFY_EMAIL_TO=you@example.com
    NOTIFY_SMTP_HOST=smtp.gmail.com
    NOTIFY_SMTP_PORT=587
    NOTIFY_SMTP_USER=you@gmail.com
    NOTIFY_SMTP_PASS=abcdefghijklmnop

Nothing here is allowed to break the run: every failure is caught and reported
on stdout, and the exit code is always 0.
"""

import argparse
import csv
import os
import socket
import sys
import urllib.error
import urllib.request

TIMEOUT = 15          # a hung network must not stall the pipeline


def training_summary(csv_path):
    """A few lines about training progress, read from the per-epoch CSV."""
    if not csv_path or not os.path.exists(csv_path):
        return []
    try:
        rows = list(csv.DictReader(open(csv_path)))
    except OSError as e:
        return [f'(could not read {csv_path}: {e})']
    if not rows:
        return ['training: no epoch finished yet']
    val = [float(r['val_loss']) for r in rows]
    best = min(val)
    return [
        f'epochs done : {rows[-1]["epoch"]}',
        f'last        : train={float(rows[-1]["train_loss"]):.4f} val={val[-1]:.4f}',
        f'best val    : {best:.4f} at epoch {val.index(best) + 1}',
    ]


def build(args):
    state = args.state.upper()
    icon = {'FINISHED': 'done', 'FAILED': 'failed', 'INTERRUPTED': 'stopped'}.get(state, state)
    title = f'[{socket.gethostname()}] SQG training {icon}'
    body = [
        f'state    : {state}',
        f'run      : {args.run}',
        f'stage    : {args.stage}',
        f'elapsed  : {args.elapsed}',
    ]
    body += training_summary(args.csv)
    if args.log:
        body += ['', f'log: {args.log}']
    if args.note:
        body += ['', args.note]
    return title, '\n'.join(body)


def send_ntfy(topic, title, body):
    req = urllib.request.Request(
        f'https://ntfy.sh/{topic}',
        data=body.encode(),
        headers={'Title': title, 'Priority': 'default'},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return f'ntfy: HTTP {r.status}'


def send_email(title, body):
    import smtplib
    from email.message import EmailMessage

    host = os.environ.get('NOTIFY_SMTP_HOST', 'smtp.gmail.com')
    port = int(os.environ.get('NOTIFY_SMTP_PORT', '587'))
    user = os.environ['NOTIFY_SMTP_USER']
    pw   = os.environ['NOTIFY_SMTP_PASS']
    to   = os.environ['NOTIFY_EMAIL_TO']

    msg = EmailMessage()
    msg['Subject'] = title
    msg['From'] = user
    msg['To'] = to
    msg.set_content(body)

    if port == 465:
        with smtplib.SMTP_SSL(host, port, timeout=TIMEOUT) as s:
            s.login(user, pw)
            s.send_message(msg)
    else:
        with smtplib.SMTP(host, port, timeout=TIMEOUT) as s:
            s.starttls()
            s.login(user, pw)
            s.send_message(msg)
    return f'email: sent to {to}'


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--state', required=True,
                   help='finished | failed | interrupted')
    p.add_argument('--run', default='?')
    p.add_argument('--stage', default='?')
    p.add_argument('--elapsed', default='?')
    p.add_argument('--log', default='')
    p.add_argument('--csv', default='')
    p.add_argument('--note', default='')
    p.add_argument('--dry-run', action='store_true',
                   help='print the message and the channels, send nothing')
    args = p.parse_args()

    title, body = build(args)

    topic = os.environ.get('NOTIFY_NTFY_TOPIC')
    has_email = all(os.environ.get(k) for k in
                    ('NOTIFY_EMAIL_TO', 'NOTIFY_SMTP_USER', 'NOTIFY_SMTP_PASS'))

    if args.dry_run:
        print(f'--- title ---\n{title}\n--- body ---\n{body}\n--- channels ---')
        print(f'ntfy : {"topic " + topic if topic else "not configured"}')
        print(f'email: {"to " + os.environ["NOTIFY_EMAIL_TO"] if has_email else "not configured"}')
        return

    if not topic and not has_email:
        print('notify: no channel configured (see tools/notify.py docstring); skipped')
        return

    results = []
    if topic:
        try:
            results.append(send_ntfy(topic, title, body))
        except (urllib.error.URLError, OSError, ValueError) as e:
            results.append(f'ntfy FAILED: {type(e).__name__}: {e}')
    if has_email:
        try:
            results.append(send_email(title, body))
        except Exception as e:                      # smtplib raises many types
            results.append(f'email FAILED: {type(e).__name__}: {e}')
    for r in results:
        print(f'notify: {r}')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:                          # never fail the pipeline
        print(f'notify: unexpected error, ignored: {type(e).__name__}: {e}')
    sys.exit(0)
