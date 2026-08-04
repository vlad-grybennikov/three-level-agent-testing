"use client";

// Read-only view of the environment snapshot (orders, appointments, events).
// Pure presentation: all data comes from GET /state on the Python server.
// onPick (optional): called with a name/email the user clicked, so the demo
// operator can insert it into the message box without typing.
export default function StatePanel({ state, onPick }) {
  if (!state) {
    return (
      <aside>
        <h2>Environment</h2>
        <p className="note">backend unreachable, start the Python server</p>
      </aside>
    );
  }

  const who = {};
  for (const s of state.subscribers) who[s.id] = s.full_name.split(" ")[0];
  const planCode = {};
  for (const p of state.plans) planCode[p.id] = p.code;
  const orderById = {};
  for (const o of state.orders) orderById[o.id] = o;

  const visitDate = (a) =>
    a.slot_id && state.slots[a.slot_id]
      ? state.slots[a.slot_id].slice(0, 16).replace("T", " ")
      : "—";

  const pick = (text) => onPick && onPick(text);

  return (
    <aside>
      <h2>Subscribers</h2>
      <table>
        <tbody>
          {state.subscribers.map((s) => (
            <tr key={s.id}>
              <td>
                <a
                  className="pick"
                  title="insert into message"
                  onClick={() => pick(s.full_name)}
                >
                  {s.full_name}
                </a>
              </td>
              <td>
                <a
                  className="pick"
                  title="insert into message"
                  onClick={() => pick(s.email)}
                >
                  {s.email}
                </a>
              </td>
              <td>
                <span className={`pill ${s.state}`}>{s.state}</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <h2>Orders</h2>
      <table>
        <tbody>
          {state.orders.map((o) => (
            <tr key={o.id}>
              <td>{o.id}</td>
              <td>{who[o.subscriber_id] ?? "?"}</td>
              <td>{planCode[o.plan_id] ?? o.plan_id}</td>
              <td>
                <span className={`pill ${o.state}`}>{o.state}</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <h2>Appointments</h2>
      <table>
        <tbody>
          {state.appointments.map((a) => (
            <tr key={a.id}>
              <td>{a.id}</td>
              <td>{who[orderById[a.order_id]?.subscriber_id] ?? "?"}</td>
              <td>{a.kind}</td>
              <td>{visitDate(a)}</td>
              <td>
                <span className={`pill ${a.status}`}>{a.status}</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <h2>
        Events log <span className="pill">{state.events.length}</span>
      </h2>
      <table>
        <tbody>
          {state.events.map((e) => (
            <tr key={e.seq}>
              <td>{e.seq}</td>
              <td>{e.operation}</td>
              <td>{e.entity_id ?? ""}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </aside>
  );
}
