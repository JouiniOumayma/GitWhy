import React from 'react'

/** @typedef {React.FC<{title: string, value: number, accent: string, icon?: string}>} StatCardProps */

export const StatCard = ({ title, value, accent, icon }) => {
  return (
    <div className="stat-card">
      <div className="stat-icon">
        {icon && <span>{icon}</span>}
      </div>
      <div className="stat-content">
        <p className="stat-title">{title}</p>
        <p className="stat-value">{value}</p>
      </div>
    </div>
  )
}