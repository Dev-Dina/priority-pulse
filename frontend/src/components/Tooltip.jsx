export default function Tooltip({ text }) {
  return (
    <span className="tt-anchor">
      <span className="tt-icon">?</span>
      <span className="tt-box">{text}</span>
    </span>
  )
}
