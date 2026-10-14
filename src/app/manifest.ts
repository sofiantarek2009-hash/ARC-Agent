import { MetadataRoute } from 'next';

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: 'NorthStar Energy Solutions',
    short_name: 'NorthStar',
    description: 'Integrated Technical Consultation, Simulation Manufacturing, & Asset Integrity Solutions for the Global Energy Sector.',
    start_url: '/',
    display: 'standalone',
    background_color: '#0B0C0E',
    theme_color: '#0B0C0E',
    icons: [
      {
        src: '/favicon.ico',
        sizes: 'any',
        type: 'image/x-icon',
      },
    ],
  };
}
