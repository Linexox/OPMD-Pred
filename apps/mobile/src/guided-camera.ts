import { Capacitor, registerPlugin } from '@capacitor/core'

export type GuidedPhoto = { path: string; format: 'jpg' }
const GuidedCamera = registerPlugin<{ takePhoto(): Promise<GuidedPhoto> }>('GuidedCamera')

/** Persist the draft before calling. Pass the result to the existing acceptCamera handler. */
export async function takeGuidedPhoto(): Promise<GuidedPhoto> {
  if (Capacitor.getPlatform() !== 'android') throw new Error('请使用系统相机或相册')
  return GuidedCamera.takePhoto()
}
