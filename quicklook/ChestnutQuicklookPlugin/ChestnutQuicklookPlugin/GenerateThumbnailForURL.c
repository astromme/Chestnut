#import <CoreFoundation/CoreFoundation.h>
#import <CoreServices/CoreServices.h>
#import <QuickLook/QuickLook.h>

/* -----------------------------------------------------------------------------
    Generate a thumbnail for file

   This function's job is to create thumbnail for designated file as fast as possible
   ----------------------------------------------------------------------------- */

OSStatus GenerateThumbnailForURL(void *thisInterface, QLThumbnailRequestRef thumbnail, CFURLRef url, CFStringRef contentTypeUTI, CFDictionaryRef options, CGSize maxSize)
{
    //#warning To complete your generator please implement the function GenerateThumbnailForURL in GenerateThumbnailForURL.c
    // not implemented yet
    
    return noErr;
}

void CancelThumbnailGeneration(void* thisInterface, QLThumbnailRequestRef thumbnail)
{
    // implement only if supported
}
